def set_params(eps: float, gamma: float, no_bias: bool) -> list:
    variants = ['zero', 'eps', 'gamma', 'komposition', 'plus']
    params = dict()

    for variant in variants:
        if variant == 'zero':
            params[variant] = ['LRP-0', {'eps': 0, 'gamma': 0, 'no_bias': no_bias, 'komp': False, 'z_pos': False}]

        elif variant == 'eps':
            params[variant] = [f'LRP-ε (ε={eps} * std)', {'eps': eps, 'gamma': 0, 'no_bias': no_bias, 'komp': False,
                                                          'z_pos': False}]

        elif variant == 'gamma':
            params[variant] = [f'LRP-γ (γ={gamma})', {'eps': 0, 'gamma': gamma, 'no_bias': no_bias, 'komp': False,
                                                      'z_pos': False}]

        elif variant == 'komposition':
            params[variant] = [f'LRP-Komposition', {'eps': 2*eps, 'gamma': 2*gamma, 'no_bias': no_bias, 'komp': True,
                                                    'z_pos': False}]

        elif variant == 'plus':
            params[variant] = [r'$z^+$', {'eps': 0, 'gamma': 0, 'no_bias': no_bias, 'komp': True, 'z_pos': True}]

    return params


def preprocess_for_lrp(model, image, mask):
    # Originalbild wird wie beim Training des Netzes angepasst
    image = preprocess_input(image.copy())

    # Kopie des Models wird angefertigt, damit Gewichte durch Funktion forward() nicht für nachfolgende Anwendungen
    # verändert werden. Letzte Aktivierung (Sigmoid) wird gelöscht.
    new_model = tf.keras.models.clone_model(model)
    new_model.pop()
    new_model.set_weights(model.get_weights())

    # Schichten des Modells werden in Array gespeichert
    layers = new_model.layers

    # Input wird in Netz gegeben und der Output jeder Schicht wird in Array gespeichert
    outputs = [image]
    for i, layer in enumerate(layers):
        output = layer(outputs[i])

        # Auf letzten output soll keine ReLU angewandt werden
        if i < len(layers) - 1 and \
                (isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense)):

            output = tf.keras.activations.relu(output)

        outputs.append(output)

    # Output des Netzes wird mit Maske multipliziert, um nur zu erklärenden Output zu erhalten
    output_const = tf.constant(outputs[-1])
    output_const = output_const * mask

    return outputs, layers, output_const
