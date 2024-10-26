import logging
import time
import torch.utils.data as data
from ema import EMA
from model import *
from torch.nn.functional import interpolate

from pretraining.Resnet import FPN as AuxCls
from pretraining.Resnet import BasicBlock,Bottleneck,BasicBlockCBAM
from pretraining.predictor import predictor
from pretraining.gumbelmodule import GumbleSoftmax
from torchvision.models import resnet18

from utils import *
from diffusion_utils import *
from tqdm import tqdm

plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.resize_ratio = [224,192,168,112]
        self.device = device
        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.test_num_timesteps = config.diffusion.test_timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start,
                                   end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)  #
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(
            dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()

        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()


        if config.diffusion.apply_aux_cls:

            self.cond_pred_model = AuxCls(block=BasicBlockCBAM, layers=[2,2,2,2]).to(self.device)
            self.resolu_model = predictor(layer_config=[1, 1, 1, 1], num_bits=4).to(self.device)


            self.gumbel_softmax = GumbleSoftmax()
            self.aux_cost_function = nn.CrossEntropyLoss()
        else:
            pass
        self.tuned_scale_T = None

    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if self.config.model.arch == "simple" or \
                (self.config.model.arch == "linear" and self.config.data.dataset == "MNIST"):
            x = torch.flatten(x, 1)
        predictor_ratio_score = self.resolu_model(x.to(self.device))
        predictor_ratio_score_gumbel = self.gumbel_softmax(predictor_ratio_score)  # one-hot 形式， 例如[1,0,0] 表示224，224
        output = 0
        for j, r in enumerate(self.resize_ratio):
            new_images = interpolate(x.to(self.device), size=(r, r))
            new_output, y_single = self.cond_pred_model(new_images, int(j))
            output += predictor_ratio_score_gumbel[:, j:j + 1] * new_output  # 只得到对应分辨率的类别预测
        top1_output = output
        y_pred = top1_output
        return y_pred, y_single

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            x_batch, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            predictor_ratio_score = self.resolu_model(x_batch.to(self.device))
            predictor_ratio_score_gumbel = self.gumbel_softmax(predictor_ratio_score)
            output = 0
            for j, r in enumerate(self.resize_ratio):
                new_images = interpolate(x_batch.to(self.device), size=(r, r))
                new_output,_ = self.cond_pred_model(new_images, int(j))
                output += predictor_ratio_score_gumbel[:, j:j + 1] * new_output
            top1_output = output
            y_pred_prob = top1_output

            y_pred_prob = y_pred_prob.softmax(dim=1)
            y_pred_label = torch.argmax(y_pred_prob, 1, keepdim=True).cpu().detach().numpy()
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer, resolu_optimizer,resize_ratio):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        predictor_input = interpolate(x_batch, size=(128, 128))
        predictor_ratio_score = self.resolu_model(x_batch)
        predictor_ratio_score_gumbel = self.gumbel_softmax(predictor_ratio_score)
        output = 0
        for j, r in enumerate(resize_ratio):
            new_images = interpolate(x_batch, size=(r, r))
            new_output,_ = self.cond_pred_model(new_images, int(j))
            output += predictor_ratio_score_gumbel[:, j:j + 1] * new_output
        top1_output = output

        aux_cost = self.aux_cost_function(top1_output, y_batch)
        aux_optimizer.zero_grad()
        resolu_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        resolu_optimizer.step()
        return aux_cost.cpu().item()

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        print('loading dataset..')

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        logging.info(f'successfully load, length of train_dataset: {len(train_dataset)}, length of test_dataset:{len(test_dataset)}')
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)
        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("\nBefore training, the guidance classifier accuracy on the test set is {:.8f}.\n\n".format(
            y_acc_aux_model))

        optimizer = get_optimizer(self.config.optim, model.parameters())
        criterion = nn.CrossEntropyLoss()
        brier_score = nn.MSELoss()

        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())
            resolu_optimizer = get_optimizer(self.config.aux_optim,
                                          self.resolu_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                self.resolu_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        if config.data.dataset == "gaussian_mixture":
                            x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                        else:
                            x_batch, y_labels_batch = feature_label_set
                            y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                                  config)
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer,
                                                                            resolu_optimizer,
                                                                            self.resize_ratio)
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                    (pretrain_end_time - pretrain_start_time) / 60))

                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                resolu_states = [
                    self.resolu_model.state_dict(),
                    resolu_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                torch.save(resolu_states, os.path.join(self.args.log_path, "resolu_ckpt.pth"))

            y_acc_aux_model = self.evaluate_guidance_model(train_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                y_acc_aux_model))
            y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
                y_acc_aux_model))

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])

                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    resolu_states = torch.load(os.path.join(self.args.log_path, "resolu_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])
                    self.resolu_model.load_state_dict(resolu_states[0])
                    resolu_optimizer.load_state_dict(resolu_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    if config.data.dataset == "gaussian_mixture":
                        x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                    else:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)

                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)

                    x_unflat_batch = x_batch.to(self.device)
                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch = torch.flatten(x_batch, 1)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    self.resolu_model.eval()
                    step += 1


                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    num_layers = 3
                    mapped_t = torch.floor((t / self.num_timesteps) * num_layers)


                    x_batch = x_batch.to(self.device)
                    y_0_batch = y_logits_batch.to(self.device)

                    y_0_hat_batch, y_0_single = self.compute_guiding_prediction(x_unflat_batch)

                    y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                    y_0_single = y_0_single.softmax(dim=1)


                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)

                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)                     
                    y_t_batch_single = q_sample(y_0_batch, y_0_single,
                                                self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e,
                                                SingleGranularities=True, mapped_t=mapped_t)

                    output = model(x_batch, y_t_batch, t, y_0_hat_batch)
                    output_single = model(x_batch, y_t_batch_single, t, y_0_single, mapped_t=mapped_t)


                    loss = ((e - output).square().mean() + 0.5*((e - output_single).square().mean()))
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = y_0_reparam_batch
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (i + 1)}"
                            )
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint train aux classifier along with diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        self.resolu_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer,
                                                                            resolu_optimizer,
                                                                            self.resize_ratio)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance auxiliary classifier joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            resolu_states = [
                                self.resolu_model.state_dict(),
                                resolu_optimizer.state_dict(),
                            ]
                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                                torch.save(
                                    resolu_states,
                                    os.path.join(self.args.log_path, "resolu_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                            torch.save(resolu_states, os.path.join(self.args.log_path, "resolu_ckpt.pth"))

                    data_start = time.time()

                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )

                # Evaluate
                if epoch % self.config.training.validation_freq == 0 \
                        or epoch + 1 == self.config.training.n_epochs:
                    model.eval()
                    self.cond_pred_model.eval()
                    self.resolu_model.eval()
                    acc_avg = 0.
                    kappa_avg = 0.
                    y1_true = None
                    y1_pred = None
                    for test_batch_idx, (images, target) in enumerate(test_loader):
                        images_unflat = images.to(self.device)
                        if config.data.dataset == "toy" \
                                or config.model.arch == "simple" \
                                or config.model.arch == "linear":
                            images = torch.flatten(images, 1)
                        images = images.to(self.device)
                        target = target.to(self.device)
                        with torch.no_grad():
                            predictor_ratio_score = self.resolu_model(images_unflat)
                            predictor_ratio_score_gumbel = self.gumbel_softmax(predictor_ratio_score)
                            output = 0
                            for j, r in enumerate(self.resize_ratio):
                                new_images = interpolate(images_unflat, size=(r, r))
                                new_output, _ = self.cond_pred_model(new_images, int(j))
                                output += predictor_ratio_score_gumbel[:, j:j + 1] * new_output
                            target_pred = output
                            target_pred = target_pred.softmax(dim=1)
                            # prior mean at timestep T
                            y_T_mean = target_pred
                            if config.diffusion.noise_prior:
                                y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                            if not config.diffusion.noise_prior:
                                target_pred,_ = self.compute_guiding_prediction(images_unflat)
                                target_pred = target_pred.softmax(dim=1)

                            label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                      self.num_timesteps, self.alphas,
                                                      self.one_minus_alphas_bar_sqrt,
                                                      only_last_sample=True)
                            y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                            y1_true = torch.cat([y1_true, target]) if y1_true is not None else target
                            acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                    kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()
                    f1_avg = compute_f1_score(y1_true, y1_pred).item()
                    recall_avg = compute_recall(y1_true, y1_pred)

                    acc_avg /= (test_batch_idx + 1)
                    if acc_avg > max_accuracy:
                        logging.info("Update best accuracy at Epoch {}.".format(epoch))
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        aux_states = [
                            self.cond_pred_model.state_dict(),
                            aux_optimizer.state_dict(),
                        ]
                        resolu_states = [
                            self.resolu_model.state_dict(),
                            resolu_optimizer.state_dict(),
                        ]
                        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt_best.pth"))
                        torch.save(resolu_states, os.path.join(self.args.log_path, "resolu_ckpt_best.pth"))
                    max_accuracy = max(max_accuracy, acc_avg)
                    if not tb_logger is None:
                        tb_logger.add_scalar('accuracy', acc_avg, global_step=step)
                    class_metrics = Get_classification_report(y1_true, y1_pred)
                    logging.info(
                        (
                                f"epoch: {epoch}, step: {step}, " +
                                f"Average accuracy: {acc_avg} Average F1: {f1_avg}, Average Recall:{recall_avg}" +
                                f"Max accuracy: {max_accuracy:.2f}%"
                        )
                    )
                    logging.info(f"\n  {class_metrics}")

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                resolu_states = [
                    self.resolu_model.state_dict(),
                    resolu_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                torch.save(resolu_states, os.path.join(self.args.log_path, "resolu_ckpt.pth"))
                # report training set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                logging.info("After joint-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                    y_acc_aux_model))
                # report test set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                logging.info("After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
                    y_acc_aux_model))

    def test(self):
        args = self.args
        config = self.config
        data_object, train_dataset, test_dataset = get_dataset(args, config)
        log_path = os.path.join(self.args.log_path)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt_best.pth"),
                                        map_location=self.device)
                resolu_states = torch.load(os.path.join(aux_cls_path, "resolu_ckpt_best.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=False)
                self.resolu_model.load_state_dict(resolu_states[0], strict=False)
                logging.info(f"Loading from: {aux_cls_path}/aux_ckpt_best.pth")
                logging.info(f"Loading from: {aux_cls_path}/resolu_ckpt_best.pth")

        # Evaluate
        model.eval()
        self.cond_pred_model.eval()
        self.resolu_model.eval()
        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
            y_acc_aux_model))
        logging.info(f"Current test_num_timesteps: {self.test_num_timesteps}")

        acc_avg = 0.
        y1_true = None
        y1_pred = None
        for test_batch_idx, (images, target) in enumerate(test_loader):
            images_unflat = images.to(self.device)
            images = images.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                predictor_ratio_score = self.resolu_model(images_unflat)
                predictor_ratio_score_gumbel = self.gumbel_softmax(predictor_ratio_score)
                output = 0
                for j, r in enumerate(self.resize_ratio):
                    new_images = interpolate(images_unflat, size=(r, r))
                    new_output,_ = self.cond_pred_model(new_images, int(j))
                    output += predictor_ratio_score_gumbel[:, j:j + 1] * new_output
                top1_output = output
                target_pred = top1_output

                target_pred = target_pred.softmax(dim=1)

                y_T_mean = target_pred
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                    target_pred,_ = self.compute_guiding_prediction(images_unflat)
                    target_pred = target_pred.softmax(dim=1)

                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                          self.test_num_timesteps, self.alphas,
                                          self.one_minus_alphas_bar_sqrt,
                                          only_last_sample=False)

                label_t_0 = label_t_0.softmax(dim=-1)
                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                y1_true = torch.cat([y1_true, target]) if y1_true is not None else target

        f1_avg = compute_f1_score(y1_true, y1_pred)
        recall_avg = compute_recall(y1_true, y1_pred)
        class_metrics = Get_classification_report(y1_true, y1_pred)
        plot_and_save_confusion_matrix(y1_true, y1_pred, save_path=log_path)
        logging.info(f"Successfully save confusion-matrix :{log_path}")
        acc_avg /= (test_batch_idx + 1)
        logging.info(
            (
                    f"[Test:] Average accuracy: {acc_avg} F1: {f1_avg}, Recall:{recall_avg} "
            )
        )
        logging.info(f"\n  {class_metrics}")