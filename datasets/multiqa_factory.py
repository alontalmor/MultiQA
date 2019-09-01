

class MultiQAFactory:
    def __init__(self):
        pass

    def build_dataset(self, dastaset_name, split, dataset_version, dataset_flavor, \
                      dataset_specific_props, preprocessor, sample_size, max_contexts_in_file, custom_input_file):


        mod = __import__('datasets.' + dastaset_name + '.' + dastaset_name.lower(), fromlist=[dastaset_name])
        dataset_class = getattr(mod, dastaset_name)(preprocessor, split, dataset_version, \
                                            dataset_flavor, dataset_specific_props, sample_size, max_contexts_in_file, custom_input_file)

        for contexts in dataset_class.build_contexts():
            header = dataset_class.build_header(contexts)
            yield header, contexts

    def format_predictions(self, dastaset_name, predictions):

        mod = __import__('datasets.' + dastaset_name + '.' + dastaset_name.lower(), fromlist=[dastaset_name])
        dataset_class = getattr(mod, dastaset_name)()

        predictions = dataset_class.format_predictions(predictions)

        return predictions



