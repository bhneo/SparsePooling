import tensorflow as tf


def get_signed_grad(input_image, input_label, pretrained_model, loss_object, target=None):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        if target is None:
            loss = loss_object(input_label, prediction)
        else:
            loss = -loss_object(target, prediction)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def fgsm(images, labels, model, loss,
         epsilon,
         x_min=None, x_max=None, target=None):
    perturbations = get_signed_grad(images, labels, model, loss, target)
    adv_x = images + epsilon * perturbations
    if x_min is not None and x_max is not None:
        adv_x = tf.clip_by_value(adv_x, x_min, x_max)
    return adv_x


def bim(images, labels, model, loss,
        epsilon, steps,
        x_min=None, x_max=None, target=None, pgd=False):
    eps_step = epsilon / steps
    if pgd:
        eta = tf.random.uniform(tf.shape(images), -epsilon, epsilon)
    else:
        eta = tf.zeros_like(images)
    for i in range(steps):
        adv_tmp = eta + images
        perturbations = get_signed_grad(adv_tmp, labels, model, loss, target)
        eta += eps_step * perturbations
        eta = tf.clip_by_value(eta, -epsilon, epsilon)
    adv_x = images + eta
    if x_min is not None and x_max is not None:
        adv_x = tf.clip_by_value(adv_x, x_min, x_max)
    return adv_x


def create_adversarial_samples(epsilon, images, labels, model, loss,
                               method='FGSM', steps=1,
                               x_min=None, x_max=None, target=None, label_sparse=True):
    if epsilon == 0:
        return images
    if not label_sparse and target is not None:
        num_out = tf.shape(labels)[-1]
        target = tf.one_hot(target, num_out)
    if method == 'FGSM':
        adv_x = fgsm(images, labels, model, loss, epsilon, x_min, x_max, target)
    elif method == 'BIM':
        adv_x = bim(images, labels, model, loss, epsilon, steps, x_min, x_max, target)
    elif method == 'PGD':
        adv_x = bim(images, labels, model, loss, epsilon, steps, x_min, x_max, target, True)
    else:
        adv_x = images
    return adv_x


def prediction_after_attack(epsilon, images, labels, model, model_src, loss,
                            method='FGSM', steps=1,
                            x_min=None, x_max=None, target=None, label_sparse=True):
    if epsilon == 0:
        adv_x = images
        prediction_adv = model(adv_x)
        return prediction_adv
    if not label_sparse and target is not None:
        num_out = tf.shape(labels)[-1]
        target = tf.one_hot(target, num_out)
    if method == 'FGSM':
        adv_x = fgsm(images, labels, model_src, loss, epsilon, x_min, x_max, target)
    elif method == 'BIM':
        adv_x = bim(images, labels, model_src, loss, epsilon, steps, x_min, x_max, target)
    elif method == 'PGD':
        adv_x = bim(images, labels, model_src, loss, epsilon, steps, x_min, x_max, target, True)
    else:
        adv_x = images
    prediction_adv = model(adv_x)
    return prediction_adv


def evaluate_attacks_success_rate(epsilons, dataset, model, loss,
                                  method='FGSM', steps=1, model_src=None,
                                  x_min=None, x_max=None, target=None,
                                  only_correct=True, label_sparse=True, cost=False):
    results = []
    import time
    if label_sparse:
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
    else:
        metric = tf.keras.metrics.CategoricalAccuracy()

    if model_src is None:
        model_src = model

    @tf.function
    def process(images, labels, target, epsilon):
        if only_correct:
            correct, images, labels, predictions = get_correct(model, images, labels, label_sparse)
            if target is not None:
                images, labels, target = filter_target(target, correct, predictions, images, labels, label_sparse)
            if tf.shape(labels)[0] > 0:
                prediction_adv = prediction_after_attack(epsilon, images, labels, model, model_src, loss, method, steps, x_min,
                                                         x_max, target)
                metric.update_state(labels, prediction_adv)

    for epsilon in epsilons:
        metric.reset_states()
        t1 = time.time()
        progress = 0
        for images, labels in dataset:
            process(images, labels, target, epsilon)
            if cost:
                progress += labels.get_shape().as_list()[0]
                print("\r{}".format(progress), end='')
        t2 = time.time()
        if cost:
            print('\ncost:', t2 - t1)
        result = 1 - metric.result()
        results.append(round(result.numpy(), 4))
    print('success_rate:', results)


def evaluate_attacks_success_rate_all_target(epsilons, dataset, model, loss, categories,
                                             method='FGSM', steps=1,
                                             x_min=None, x_max=None, model_src=None,
                                             only_correct=True, label_sparse=True, cost=False):
    if label_sparse:
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
    else:
        metric = tf.keras.metrics.CategoricalAccuracy()

    if model_src is None:
        model_src = model

    # @tf.function
    def process(images, labels, ori, target, epsilon):
        if label_sparse:
            idx_ori = tf.equal(labels, ori)
        else:
            sparse = tf.argmax(labels, -1)
            idx_ori = tf.equal(sparse, ori)
        images = images[idx_ori]
        labels = labels[idx_ori]
        if only_correct:
            correct, images, labels, predictions = get_correct(model, images, labels, label_sparse)
        if tf.shape(labels)[0] > 0:
            target_arr = tf.tile(tf.constant([target]), [len(labels)])
            prediction_adv = prediction_after_attack(epsilon=epsilon, images=images, labels=labels,
                                                     model=model, model_src=model_src,
                                                     loss=loss, method=method, steps=steps,
                                                     x_min=x_min, x_max=x_max,
                                                     target=target_arr, label_sparse=label_sparse)
            metric.update_state(target_arr, prediction_adv)

    for epsilon in epsilons:
        print('eps:', epsilon)
        for ori in categories:
            results = []
            for target in categories:
                metric.reset_states()
                progress = 0
                for images, labels in dataset:
                    process(images, labels, ori, target, epsilon)
                    if cost:
                        progress += labels.get_shape().as_list()[0]
                        print("\r{}".format(progress), end='')
                result = metric.result()
                results.append(round(result.numpy(), 4))
            print(str(ori), ':', results)


def evaluate_model_after_attacks(epsilons, metric, dataset, model, loss,
                                 method='FGSM', steps=1, model_src=None,
                                 x_min=None, x_max=None, label_sparse=True, cost=False):
    results = []
    import time

    if model_src is None:
        model_src = model

    @tf.function
    def process(images, labels, epsilon):
        adv_x = create_adversarial_samples(epsilon, images, labels, model_src, loss, method, steps, x_min, x_max)
        prediction_adv = model(adv_x)
        metric.update_state(labels, prediction_adv)

    for epsilon in epsilons:
        metric.reset_states()
        t1 = time.time()
        progress = 0
        for images, labels in dataset:
            process(images, labels, epsilon)
            if cost:
                progress+=labels.get_shape().as_list()[0]
                print("\r{}".format(progress), end='')
        t2 = time.time()
        if cost:
            print('\ncost:', t2 - t1)
        result = metric.result()
        results.append(round(result.numpy(), 4))
    print('results:', results)


def get_correct(model, images, labels, label_sparse=True):
    if label_sparse is False:
        labels = tf.argmax(labels, -1)
    predictions = model(images)
    num_out = predictions.get_shape().as_list()[-1]
    predictions = tf.argmax(predictions, -1)
    correct = tf.equal(predictions, labels)
    images = images[correct]
    labels = labels[correct]
    predictions = predictions[correct]
    if label_sparse is False:
        labels = tf.one_hot(labels, num_out)
    return correct, images, labels, predictions


def filter_target(target, correct, predictions, images, labels, label_sparse=True):
    num_out = 10
    if label_sparse is False:
        num_out = labels.get_shape().as_list()[-1]
        labels = tf.argmax(labels, -1)
    target = target[correct]
    not_target = tf.not_equal(predictions, target)
    images = images[not_target]
    labels = labels[not_target]
    target = target[not_target]
    if label_sparse is False:
        labels = tf.one_hot(labels, num_out)
    return images, labels, target