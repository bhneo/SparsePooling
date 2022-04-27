import tensorflow as tf
import numpy as np
import os

from common.inputs.voc2010 import voc_parts


class TensorLog(object):
    def __init__(self):
        self.hist = {}
        self.scalar = {}
        self.image = {}
        self.tensor = {}
        self.model = None

    def add_hist(self, name, tensor):
        self.hist[name] = tensor

    def add_scalar(self, name, tensor):
        self.scalar[name] = tensor

    def add_image(self, name, image):
        self.image[name] = image

    def add_tensor(self, name, tensor):
        self.tensor[name] = tensor

    def get_outputs(self):
        outputs = []
        for key in self.hist:
            outputs.append(self.hist[key])
        for key in self.scalar:
            outputs.append(self.scalar[key])
        for key in self.image:
            outputs.append(self.image[key])
        for key in self.tensor:
            outputs.append(self.tensor[key])
        return outputs

    def set_model(self, model):
        self.model = model

    def summary(self, outputs, epoch):
        i = 0
        for key in self.hist:
            tf.summary.histogram(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.scalar:
            tf.summary.scalar(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.image:
            tf.summary.image(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.tensor:
            i += 1


def init_devices(memory_growth=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, memory_growth)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_distribute_strategy(batch_size_per_replica, strategy_name='mirror'):
    if "mirror" in strategy_name.lower():
        strategy = tf.distribute.MirroredStrategy()
    elif "multi_worker" in strategy_name.lower():
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        print('unknown strategy!')
        strategy = None

    if strategy is None:
        global_batch_size = batch_size_per_replica
    else:
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    return strategy, global_batch_size


def str2bool(obj):
    if isinstance(obj, str):
        if obj == 'True':
            return True
        elif obj == 'False':
            return False
        elif obj == 'None':
            return None
        return obj
    if isinstance(obj, bool):
        return obj
    else:
        raise TypeError('{} is not str'.format(obj))


def get_cam(feature_conv, fc_weight, class_idx, height, width):
    # Keras default is channels last, hence nc is in last
    feature_conv = feature_conv.numpy()
    bz, h, w, nc = feature_conv.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        cam = np.dot(fc_weight[:, idx], np.transpose(feature_conv[i].reshape(h * w, nc)))
        cam = cam.reshape((h, w, 1))
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        output_cam.append(tf.image.resize(cam_img, (height, width)).numpy())
    return output_cam


def get_caps_cam(feature_conv, c, class_idx, height, width, target=2, sub=0, weight=None):
    feature_conv = feature_conv.numpy()
    if not isinstance(c, np.ndarray):
        c = c.numpy()
    bz, h, w, nc = feature_conv.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        if isinstance(idx, np.int64):
            importance = c[i, :, idx, 0].reshape([1, 1, -1])
            if weight is not None:
                activation = weight[i].numpy().reshape([1, 1, -1])
                importance *= activation
        else:
            importance = []
            top_k_idx = idx.argsort()[::-1][0:target]
            for idx_i in top_k_idx:
                importance.append(c[i, :, idx_i, 0].reshape([1, 1, -1]))
            importance = np.concatenate(importance, axis=0)
            importance = np.sum(importance, axis=0, keepdims=True)
        cam = np.sum(feature_conv[i] * importance, axis=-1, keepdims=True)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        if sub > 0:
            sub_maps_idx = importance.reshape(-1).argsort()[::-1][0:sub]
            sub_maps = []
            for id in sub_maps_idx:
                sub_map = np.expand_dims(feature_conv[i, :, :, id], -1)
                sub_map = sub_map - np.min(sub_map)
                sub_map = sub_map / np.max(sub_map)
                sub_maps.append(sub_map)
            cam = np.concatenate([cam] + sub_maps, axis=1)
        output_cam.append(tf.image.resize(cam, (height, width*(sub+1))).numpy())
    return output_cam


def get_c_cam(c, interpret, class_idx, height, width):
    c = c.numpy()
    bs, c_in, c_out, _ = c.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        importance = c[i, :, idx*interpret:(idx+1)*interpret, 0].reshape([int(np.sqrt(c_in)), int(np.sqrt(c_in)), interpret])
        cam = np.sum(importance, axis=-1, keepdims=True)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        output_cam.append(tf.image.resize(cam_img, (height, width)).numpy())
    return output_cam


def get_c_cam2(c, child_probs, parts, class_idx, height, width):
    c = c.numpy()
    child_probs = child_probs.numpy()
    bs, c_in, c_out, _ = c.shape
    spatial = int(np.sqrt(c_in/parts))
    child_probs = child_probs.reshape((-1, spatial, spatial, parts))
    output_cam = []
    for i, idx in enumerate(class_idx):
        importance = c[i, :, idx, 0].reshape([spatial, spatial, parts])
        importance *= child_probs[i]
        cam = np.sum(importance, axis=-1, keepdims=True)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)

        output_cam.append(tf.image.resize(cam_img, (height, width)).numpy())
    return output_cam


def get_spatial_caps_viz(feature_shape, c, class_idx, height, width, child_probs=None):
    c = c.numpy()
    if child_probs is not None:
        child_probs = child_probs.numpy()
    h, w = feature_shape
    output = []
    for i, idx in enumerate(class_idx):
        importance = c[i, :, idx, 0].reshape([h, w, 1])
        if child_probs is not None:
            child_prob = child_probs[i].reshape([h, w, 1])
            importance *= child_prob
        importance = importance - np.min(importance)
        importance = importance / np.max(importance)

        output.append(tf.image.resize(importance, (height, width)).numpy())
    return output


def get_top_channel_caps_heatmap(feature_conv, c, classes, height, width, top_k=3, child_probs=None):
    feature_conv = feature_conv.numpy()
    c = c.numpy()
    if child_probs is not None:
        child_probs = child_probs.numpy()
    bz, h, w, nc = feature_conv.shape
    output_parts = []
    part_probs = []
    for i, class_idx in enumerate(classes):
        importance = c[i, :, class_idx, 0].reshape(-1)
        if child_probs is not None:
            child_prob = child_probs[i].reshape(-1)
            importance *= child_prob
        val, index = tf.math.top_k(importance, top_k)
        parts = []
        top_prob = []
        for j in range(top_k):
            part = feature_conv[i, :, :, index[j]]
            part = part - np.min(part)
            part = part / np.max(part)
            part = np.expand_dims(part, -1)
            part = tf.image.resize(part, (height, width)).numpy()
            parts.append(part)
            top_prob.append(val[j].numpy())
        output_parts.append(np.concatenate(parts, 1))
        part_probs.append(top_prob)
    return part_probs, output_parts


def get_spatial_activation(part, t, binary=True):
    part = part - t
    if binary:
        make_binary(part)
    return part


def make_binary(part):
    part[part < 0] = 0
    part[part > 0] = 1
    return part


def get_top_channel_cnn_rf(feature_conv, weights, classes, height, width, t, top_k=3):
    feature_conv = feature_conv.numpy()
    output_parts = []
    part_probs = []
    for i, class_idx in enumerate(classes):
        importance = weights[:, class_idx].reshape(-1)
        val, index = tf.math.top_k(importance, top_k)
        parts = []
        top_prob = []
        for j in range(top_k):
            part = feature_conv[i, :, :, index[j]]
            part = np.expand_dims(part, -1)
            part = tf.image.resize(part, (height, width)).numpy()
            part = get_spatial_activation(part, t=t[index[j]])
            parts.append(part)
            top_prob.append(val[j].numpy())
        output_parts.append(np.concatenate(parts, 1))
        part_probs.append(top_prob)
    return part_probs, output_parts


def get_top_channel_caps_rf(feature_conv, c, classes, t, height, width, top_k=3, child_probs=None):
    feature_conv = feature_conv.numpy()
    c = c.numpy()
    if child_probs is not None:
        child_probs = child_probs.numpy()
    bz, h, w, nc = feature_conv.shape
    output_parts = []
    part_probs = []
    for i, class_idx in enumerate(classes):
        importance = c[i, :, class_idx, 0].reshape(-1)
        if child_probs is not None:
            child_prob = child_probs[i].reshape(-1)
            importance *= child_prob
        val, index = tf.math.top_k(importance, top_k)
        parts = []
        top_prob = []
        for j in range(top_k):
            part = feature_conv[i, :, :, index[j]]
            part = np.expand_dims(part, -1)
            part = tf.image.resize(part, (height, width)).numpy()
            part = get_spatial_activation(part, t=t[index[j]])
            parts.append(part)
            top_prob.append(val[j].numpy())
        output_parts.append(np.concatenate(parts, 1))
        part_probs.append(top_prob)
    return part_probs, output_parts


def get_associate(fs, ps, shape, t, ass_t=0.2):
    fs = get_spatial_activation(fs, t, False)
    fs = tf.image.resize(fs, [shape[0], shape[1]])
    fs = fs.numpy()
    fs = make_binary(fs)

    associate_fp = []
    for f, p in zip(fs, ps):
        associates_p = []
        for k in p:
            iou = get_iou(f, p[k])[:, np.newaxis]
            iou[iou > ass_t] = 1
            iou[iou <= ass_t] = 0
            associates_p.append(iou)
        associates_p = np.concatenate(associates_p, 1)
        associate_fp.append(associates_p[np.newaxis, :])
    associate_fp = np.concatenate(associate_fp, 0)
    return associate_fp


def get_iou(f, p):
    i = f * p
    u = f + p
    u[u > 0] = 1
    i = np.sum(i, axis=(0, 1))
    u = np.sum(u, axis=(0, 1))
    result = []
    for _i, _u in zip(i, u):
        if _i == 0:
            iou = 0
        else:
            iou = _i/_u
        result.append(iou)
    result = np.array(result)
    return result


def get_threshold(feature_maps, rate):
    if isinstance(feature_maps, tf.Tensor):
        feature_maps = feature_maps.numpy()
    feature_maps = feature_maps.reshape([-1, feature_maps.shape[-1]])
    result = np.percentile(feature_maps, (1-rate)*100, axis=0)
    return result


def load_feature_map(root, test_model, data_shape, model_dir, caps=False, weight=None):
    file_name = 'feature_map.npy'
    if caps:
        file_name = 'c_feature_map.npy'
    elif weight is not None:
        file_name = 'w_feature_map.npy'
        weight = weight[np.newaxis, np.newaxis, np.newaxis, :, :]
    feature_path = os.path.join(model_dir, file_name)
    if os.path.exists(feature_path):
        result = np.load(feature_path, allow_pickle=True).item()
    else:
        batch = 16
        test_all, info = voc_parts.get_test_set_with_landmark(root + 'data', batch_size=batch, shape=data_shape)
        samples = info.splits['test_examples']
        progress = 0
        feature_maps = []
        all_labels = []
        for images, boxes, labels, masks in test_all:
            if caps:
                feature_conv, (parent_poses, parent_probs, cs) = test_model(images)
                predictions = np.argmax(parent_probs[-1], -1)
                c = cs[-1][:, np.newaxis, np.newaxis, :, :, :]
                c = np.squeeze(c, -1)
                w = []
                for i, pred in enumerate(predictions):
                    w.append(c[i, :, :, :, pred])
                w = np.stack(w, 0)
                feature_conv *= w
            else:
                feature_conv, probs = test_model(images)
                if weight is not None:
                    predictions = np.argmax(probs, -1)
                    w = []
                    for pred in predictions:
                        w.append(weight[:, :, :, :, pred])
                    w = np.concatenate(w, 0)
                    feature_conv *= w
            feature_maps.append(feature_conv.numpy())
            all_labels.append(labels.numpy())
            progress += images.get_shape().as_list()[0]
            print('progress: {}%'.format(int(100 * progress / samples)))
        feature_maps = np.concatenate(feature_maps, 0)
        all_labels = np.concatenate(all_labels, 0)
        result = {'features': feature_maps, 'labels': all_labels}
        np.save(feature_path, result)
    return result


def get_associate_score(data, data_shape, feature, t=None):
    progress = 0
    associates = []
    if t is None:
        t = get_threshold(feature, rate=0.005)
    for images, boxes, labels, masks in data:
        txt_label = [voc_parts.CATEGORIES[l] for l in labels]
        _, _, masks = voc_parts.parse_masks(images, masks, txt_label, boxes)
        parts = voc_parts.get_parts(masks)
        cur_batch = images.get_shape().as_list()[0]
        feature_conv = feature[progress: progress + cur_batch]
        progress += cur_batch
        associate_fp = get_associate(feature_conv,
                                     parts,
                                     data_shape,
                                     t=t)
        associates.append(associate_fp)

    associates = np.concatenate(associates, 0)
    associates = np.mean(associates, 0)
    associates = np.max(associates, -1)
    return associates


def get_model_dir(backbone, log='log', routing='AVG', dataset='voc2010',
                  inverted=False, target=None, iter_num=None, temper=None, atoms=None,
                  re=0.1, finetune=0, parts=128, bs=32, idx=1):
    model_dir = '{}/{}/int_{}_{}_fine{}'.format(log, dataset, routing, backbone, finetune)
    if inverted:
        model_dir += '_inv'
    if target is not None:
        model_dir += '_{}'.format(target)
    if iter_num is not None:
        model_dir += '_it{}'.format(iter_num)
    if temper is not None:
        model_dir += '_temper{}'.format(temper)
    if parts is not None:
        model_dir += '_part{}'.format(parts)
    if atoms is not None:
        model_dir += '_atom{}'.format(atoms)
    if re > 0:
        model_dir += '_re{}'.format(re)
    model_dir += '_trial{}_bs{}_flip_crop'.format(idx, bs)

    if not os.path.exists(model_dir):
        raise Exception('model not exist:{}'.format(model_dir))
    return model_dir


def get_shape(backbone):
    if backbone == 'InceptionV3':
        data_shape = (299, 299, 3)
    else:
        data_shape = (224, 224, 3)
    return data_shape


def load_belonging(root, test_model, data_shape, model_dir):
    belonging_path = os.path.join(model_dir, 'belonging.npy')
    if os.path.exists(belonging_path):
        belonging = np.load(belonging_path)
    else:
        print('Computing belongings!')
        feature_maps = load_feature_map(root, test_model, data_shape, model_dir)
        features = feature_maps['features']
        labels = feature_maps['labels']
        confidence = np.sum(features, (1, 2))
        final_map = []
        for l in range(6):
            label_index = np.argwhere(labels == l)
            _map = confidence[label_index]
            _map = np.mean(_map, 0)
            final_map.append(_map)
        final_map = np.concatenate(final_map, 0)
        belonging = np.argmax(final_map, axis=0)
        np.save(os.path.join(model_dir, 'belong.npy'), belonging)
    return belonging


def calculate_time(test_set, model, metric):
    import time
    with tf.device('/GPU:0'):
        gpu_results = []
        for image, label in test_set:
            t1 = time.time() * 1000
            result = model(image)
            t2 = time.time() * 1000
            gpu_results.append(t2 - t1)
            metric.update_state(label, result)
    mean2, var2 = tf.nn.moments(tf.constant(gpu_results[1:]), 0)
    print('GPU:', mean2.numpy(), var2.numpy())
    print('acc:', metric.result().numpy())

    metric.reset_states()
    with tf.device('/CPU:0'):
        cpu_results = []
        for image, label in test_set:
            t1 = time.time() * 1000
            result = model(image)
            t2 = time.time() * 1000
            cpu_results.append(t2 - t1)
            metric.update_state(label, result)
    mean, var = tf.nn.moments(tf.constant(cpu_results), 0)
    print('CPU:', mean.numpy(), var.numpy())
    print('acc:', metric.result().numpy())


def parse_resblock(block_str):
    blocks = []
    for block in block_str:
        blocks.append(int(block))
    return blocks


def smallest_size_at_least(height, width, resize_min):
    resize_min = tf.cast(resize_min, tf.float32)
    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim
    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)
    return new_height, new_width


def central_crop(image, crop_height, crop_width):
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def to_heatmaps(ori_img, heatmap, to_rgb=False):
    import cv2
    heatmap = np.uint8(255 * heatmap)
    img = np.uint8(255 * ori_img)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if img.shape[1] != heatmap.shape[1]:
        time = heatmap.shape[1] // img.shape[1]
        img = [img for _ in range(time)]
        img = np.concatenate(img, 1)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    if to_rgb:
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img


def plot_analysis(cs_advs, heatmaps_advs, probs_adv):
    CATEGORIES = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    advs = len(cs_advs)
    bs = len(cs_advs[0])
    for i in range(bs):
        fig = plt.figure()
        fig.tight_layout()
        # ax_o = plt.axes()
        for j in range(advs):
            # ax_i = plt.axes([0, 1/advs, 1, 1/advs])
            # plt.setp(ax_i.get_xticklabels(), visible=False)
            # ax_i.set_ylabel('Epsilon')
            gs = gridspec.GridSpec(advs, 7)
            for cat in range(7):
                ax = plt.subplot(gs[j, cat])
                if cat == 6:
                    ax.figure.figimage(heatmaps_advs[j][i])
                    continue
                c = cs_advs[j][i, :, cat, 0]
                # c *= probs_adv[j][i, cat]
                # c = -np.log(c)
                hist = sns.distplot(c.reshape(-1), kde=False, ax=ax, bins=50)
                # ax.set_xlim([0,1])
                # ax.set_ylim(ylim)
                if cat != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
                if j != advs-1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if j == 0:
                    ax.set_title(CATEGORIES[cat])
        plt.show()
    return None


def plot_analysis2(adv_heatmaps_es, probs_es, entropys_es):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    from scipy import misc
    import cv2

    import PIL
    from PIL import Image
    x, y = 0.1, 0.2
    hist_w = 0.23
    hist_h = 0.7
    # entro_w = 0.9
    # entro_h = 0.35

    his1a = [x, y, hist_w, hist_h]
    his2a = [x + hist_w + 0.026, y, hist_w, hist_h]
    his3a = [x + 2 * (hist_w + 0.026), y, hist_w, hist_h]
    his4a = [x + 3 * (hist_w + 0.026), y, hist_w, hist_h]
    # entroa = [x, y + hist_h + 0.1, entro_w, entro_h]

    for adv_heatmap_es, prob_es, entropy_es in zip(adv_heatmaps_es, probs_es, entropys_es):
        plt.figure(figsize=(12.8, 2.4))
        his1 = plt.axes(his1a)
        his1.set_ylim(0, 1)
        his1.set_xlabel('Epsilon=0')
        his2 = plt.axes(his2a)
        his2.set_yticks([])
        his2.set_ylim(0,1)
        his1.set_xlabel('Epsilon=0.1')
        his3 = plt.axes(his3a)
        his3.set_yticks([])
        his3.set_ylim(0, 1)
        his1.set_xlabel('Epsilon=0.2')
        his4 = plt.axes(his4a)
        his4.set_yticks([])
        his4.set_ylim(0, 1)
        his1.set_xlabel('Epsilon=0.3')
        # entro = plt.axes(entroa)
        x_label = ['Bird', 'Cat', 'Cow', 'Dog', 'Horse', 'Sheep']
        sns.barplot(x_label, prob_es[0], ax=his1)
        sns.barplot(x_label, prob_es[1], ax=his2)
        sns.barplot(x_label, prob_es[2], ax=his3)
        sns.barplot(x_label, prob_es[3], ax=his4)
        # sns.lineplot([0, 0.1, 0.2, 0.3], entropy_es, ax=entro)
        # plt.show()

        buffer_ = BytesIO()
        plt.savefig('x.png', format='png')
        plt.cla()
        plt.clf()
        # dataPIL = PIL.Image.open(buffer_)
        # data = np.asarray(dataPIL)
        data = cv2.imread('x.png')
        # buffer_.close()
        data = tf.image.resize(data, (200, 960))
        adv_heatmap_es = np.transpose(adv_heatmap_es, [1,0,2,3])
        shape = adv_heatmap_es.shape
        adv_heatmap_es = np.resize(adv_heatmap_es, (shape[0], shape[2]*4, shape[3]))
        heatmap = tf.image.resize(adv_heatmap_es, (240, 960))

        result = np.concatenate([heatmap, data], 0)
        plt.imshow(result/255.)
        plt.axis('off')
        plt.show()


def plot_analysis3(adv_heatmaps_es, probs_es, entropys_es, batch, save=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    x_label = ['Bird', 'Cat', 'Cow', 'Dog', 'Horse', 'Sheep']
    for k, (adv_heatmap_es, prob_es, entropy_es) in enumerate(zip(adv_heatmaps_es, probs_es, entropys_es)):
        fig = plt.figure(figsize=(13.8, 4.8))
        fig.tight_layout()
        for i in range(4):
            ax = plt.subplot2grid((5, 12), (0, i*3), rowspan=3, colspan=3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(adv_heatmap_es[i])
        for i in range(4):
            ax = plt.subplot2grid((5, 12), (3, i*3), rowspan=2, colspan=3)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Epsilon={}'.format(round(0.1*i, 1)))
            if i != 0:
                ax.set_yticks([])
            sns.barplot(x_label, prob_es[i], ax=ax)
        plt.subplots_adjust(bottom=0.1, wspace=0.25)
        if save:
            plt.savefig(save + '{}_{}.png'.format(batch, k))
        else:
            plt.show()
        plt.close()


