import random

class FileClassifier:
    def __init__(self, file_list, label_list):
        self.file_list = file_list
        self.label_list = label_list
        # Caching the indices of each label for quick access
        self.spoof_indices = [i for i, label in enumerate(label_list) if label == 'spoof']
        self.bonafide_indices = [i for i, label in enumerate(label_list) if label == 'bonafide']

    def get_files(self, idx):
        # Check provided index for 'bonafide' or 'spoof'
        label = self.label_list[idx]
        if label == 'bonafide':
            bona_files = self._get_random_files(self.bonafide_indices, idx, 1)
            spoof_files = self._get_random_files(self.spoof_indices, None, 4)
            return {
                'bona1': self.file_list[idx],  # The indexed file is bona1
                'bona2': bona_files[0],        # The additional bonafide file is bona2
                'spoof1': spoof_files[0],      # The first spoof file
                'spoof2': spoof_files[1],      # The second spoof file
                'spoof3': spoof_files[2],      # The third spoof file
                'spoof4': spoof_files[3]       # The fourth spoof file
            }
        elif label == 'spoof':
            bona_files = self._get_random_files(self.bonafide_indices, None, 2)
            spoof_files = self._get_random_files(self.spoof_indices, idx, 3)
            return {
                'spoof1': self.file_list[idx],  # The indexed file is spoof1
                'spoof2': spoof_files[0],      # The first additional spoof file
                'spoof3': spoof_files[1],      # The second additional spoof file
                'spoof4': spoof_files[2],      # The third additional spoof file
                'bona1': bona_files[0],        # The first bonafide file
                'bona2': bona_files[1]         # The second bonafide file
            }
        else:
            raise ValueError("Invalid label at index {}".format(idx))

    def _get_random_files(self, indices_list, exclude_idx, number_needed):
        if exclude_idx is not None:
            possible_indices = list(set(indices_list) - {exclude_idx})
        else:
            possible_indices = list(indices_list)
        if len(possible_indices) < number_needed:
            raise ValueError("Not enough files to select from.")
        selected_indices = random.sample(possible_indices, k=number_needed)
        return [self.file_list[idx] for idx in selected_indices]

# Example usage:
file_list = ['file1.mp3', 'file2.mp3', 'file3.mp3', 'file4.mp3', 'file5.mp3', 'file6.mp3', 'file7.mp3', 'file8.mp3']
label_list = ['spoof', 'bonafide', 'spoof', 'bonafide', 'spoof', 'bonafide', 'spoof', 'bonafide']

# Create an instance of the class with given file and label lists
classifier = FileClassifier(file_list, label_list)

# # Let's say we want to check index 1, which is 'bonafide'
# file_assignments = classifier.get_files(1)
# print(file_assignments)

# # If we want to check index 0, which is 'spoof'
# file_assignments = classifier.get_files(0)
# print(file_assignments)

# file_assignments = classifier.get_files(2)
# print(file_assignments)

file_assignments = classifier.get_files(7)
print(file_assignments)
