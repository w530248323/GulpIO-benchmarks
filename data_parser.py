import os
import csv

from collections import namedtuple

ListData = namedtuple('ListData',
                      ['label_idx',
                       'label',
                       'folder'])


class JpegDataset(object):

    def __init__(self, csv_path, data_root):
        self.data = self.read_csv(csv_path, data_root)
        self._create_label_lookup()

    def read_csv(self, csv_path, data_root):
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                print(row)
                raise ValueError


    def _create_label_lookup(self):
        self.label2id = TwoWayDict()
        for idx, label in enumerate(sorted(set(self.data_df['template']))):
            self.label2id[idx] = label
        self.classes = sorted(set(self.data_df['template']))
        self.num_classes = len(self.classes)

    def _refresh_classes(self, selected_classes):
        self.data_df = self.data_df[self.data_df['template'].isin(selected_classes)]
        # create lookup again since we might remove some classes
        self._create_label_lookup()

    def summarize(self):
        # number of instances
        print("\nNumber of instances ----------------------------------------")
        print(self.data_df.shape[0])

        # number of classes
        print("\nNumber of classes ------------------------------------------")
        print(self.data_df.template.nunique())

        # number of instances per class
        print("\nNumber of instances per class ------------------------------")
        print(self.data_df.template.value_counts())

        # max and min duration of vides
        print("\nMax, Min, Avg, Std duration --------------------------------")
        print(self.data_df.duration.max())
        print(self.data_df.duration.min())
        print(self.data_df.duration.mean())
        print(self.data_df.duration.std())

        print("\n")

    def select_classes_min_instances(self, thresh):
        """
        Remove classes with lower number of instances given the threshold
        """
        class_counts = self.data_df.template.value_counts()
        selected_classes = class_counts[class_counts>thresh].keys()
        self._refresh_classes(selected_classes)
        

    def select_top_N_classes(self, N):
        """
        Select top N classes with highest number of instance
        """
        class_counts = self.data_df.template.value_counts()
        selected_classes = class_counts.keys()[:N]
        self._refresh_classes(selected_classes)

    def group_classes_by_prefix(self):
        classes = self.data_df.template.unique()
        unique_prefixes = list(set([class_name.split(' ')[0] for class_name in classes]))
        unique_prefixes = sorted(unique_prefixes)
        print(unique_prefixes)  

        for idx, label in enumerate(unique_prefixes):
            self.label2id[idx] = label

        labels_all = self.data_df.template
        for i, class_name in enumerate(unique_prefixes):
            mask = labels_all.str.startswith(class_name)
            self.data_df['template'][mask] = class_name
        self._refresh_classes(unique_prefixes)


    def data2list(self):
        '''
        Convert data df to list of labels, labels_ids and video_paths
        '''
        # load string labels
        labels = list(self.data_df.template)
        # ignore the file names only keep folder names
        files = [os.path.join(self.data_url, os.path.dirname(file))
                 for file in list(self.data_df.file)]
        # load label idxs starting from 0 to N
        label_idxs = [self.label2id[label] for label in labels]
        # set final format
        list_data = [ListData(idx, lbl, f)
                     for idx, lbl, f in zip(label_idxs, labels, files)]
        return list_data