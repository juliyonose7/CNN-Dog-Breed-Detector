"""
Top 50 Dog Breeds Configuration
================================

This module contains the configuration data for the top 50 most represented
dog breeds in the dataset, including:

- TOP_50_BREEDS: Complete breed information with directory paths and indices
- BREED_NAME_TO_INDEX: Quick lookup from breed name to class index
- BREED_INDEX_TO_DISPLAY: Quick lookup from index to display name

Breeds are ordered by image count (descending) with Maltese having the most
images (252) and Saint Bernard having the least in top 50 (170).

Author: Dog Breed Classifier Team
Date: 2024
"""

# Configuration dictionary containing all 50 breed entries
# Each entry has: name, display_name, original_dir, image_count, class_index

TOP_50_BREEDS = {'total_breeds': 50, 'breeds': {0: {'name': 'Maltese_dog', 'display_name': 'Maltese Dog', 'original_dir': 'n02085936-Maltese_dog', 'image_count': 252, 'class_index': 0}, 1: {'name': 'Afghan_hound', 'display_name': 'Afghan Hound', 'original_dir': 'n02088094-Afghan_hound', 'image_count': 239, 'class_index': 1}, 2: {'name': 'Scottish_deerhound', 'display_name': 'Scottish Deerhound', 'original_dir': 'n02092002-Scottish_deerhound', 'image_count': 232, 'class_index': 2}, 3: {'name': 'Pomeranian', 'display_name': 'Pomeranian', 'original_dir': 'n02112018-Pomeranian', 'image_count': 219, 'class_index': 3}, 4: {'name': 'Irish_wolfhound', 'display_name': 'Irish Wolfhound', 'original_dir': 'n02090721-Irish_wolfhound', 'image_count': 218, 'class_index': 4}, 5: {'name': 'Bernese_mountain_dog', 'display_name': 'Bernese Mountain Dog', 'original_dir': 'n02107683-Bernese_mountain_dog', 'image_count': 218, 'class_index': 5}, 6: {'name': 'Samoyed', 'display_name': 'Samoyed', 'original_dir': 'n02111889-Samoyed', 'image_count': 218, 'class_index': 6}, 7: {'name': 'Shih-Tzu', 'display_name': 'Shih-Tzu', 'original_dir': 'n02086240-Shih-Tzu', 'image_count': 214, 'class_index': 7}, 8: {'name': 'Great_Pyrenees', 'display_name': 'Great Pyrenees', 'original_dir': 'n02111500-Great_Pyrenees', 'image_count': 213, 'class_index': 8}, 9: {'name': 'Leonberg', 'display_name': 'Leonberg', 'original_dir': 'n02111129-Leonberg', 'image_count': 210, 'class_index': 9}, 10: {'name': 'basenji', 'display_name': 'Basenji', 'original_dir': 'n02110806-basenji', 'image_count': 209, 'class_index': 10}, 11: {'name': 'Tibetan_terrier', 'display_name': 'Tibetan Terrier', 'original_dir': 'n02097474-Tibetan_terrier', 'image_count': 206, 'class_index': 11}, 12: {'name': 'Sealyham_terrier', 'display_name': 'Sealyham Terrier', 'original_dir': 'n02095889-Sealyham_terrier', 'image_count': 202, 'class_index': 12}, 13: {'name': 'Airedale', 'display_name': 'Airedale', 'original_dir': 'n02096051-Airedale', 'image_count': 202, 'class_index': 13}, 14: {'name': 'EntleBucher', 'display_name': 'Entlebucher', 'original_dir': 'n02108000-EntleBucher', 'image_count': 202, 'class_index': 14}, 15: {'name': 'Saluki', 'display_name': 'Saluki', 'original_dir': 'n02091831-Saluki', 'image_count': 200, 'class_index': 15}, 16: {'name': 'pug', 'display_name': 'Pug', 'original_dir': 'n02110958-pug', 'image_count': 200, 'class_index': 16}, 17: {'name': 'Lakeland_terrier', 'display_name': 'Lakeland Terrier', 'original_dir': 'n02095570-Lakeland_terrier', 'image_count': 197, 'class_index': 17}, 18: {'name': 'cairn', 'display_name': 'Cairn', 'original_dir': 'n02096177-cairn', 'image_count': 197, 'class_index': 18}, 19: {'name': 'papillon', 'display_name': 'Papillon', 'original_dir': 'n02086910-papillon', 'image_count': 196, 'class_index': 19}, 20: {'name': 'Norwegian_elkhound', 'display_name': 'Norwegian Elkhound', 'original_dir': 'n02091467-Norwegian_elkhound', 'image_count': 196, 'class_index': 20}, 21: {'name': 'Australian_terrier', 'display_name': 'Australian Terrier', 'original_dir': 'n02096294-Australian_terrier', 'image_count': 196, 'class_index': 21}, 22: {'name': 'chow', 'display_name': 'Chow', 'original_dir': 'n02112137-chow', 'image_count': 196, 'class_index': 22}, 23: {'name': 'beagle', 'display_name': 'Beagle', 'original_dir': 'n02088364-beagle', 'image_count': 195, 'class_index': 23}, 24: {'name': 'Newfoundland', 'display_name': 'Newfoundland', 'original_dir': 'n02111277-Newfoundland', 'image_count': 195, 'class_index': 24}, 25: {'name': 'Siberian_husky', 'display_name': 'Siberian Husky', 'original_dir': 'n02110185-Siberian_husky', 'image_count': 192, 'class_index': 25}, 26: {'name': 'Blenheim_spaniel', 'display_name': 'Blenheim Spaniel', 'original_dir': 'n02086646-Blenheim_spaniel', 'image_count': 188, 'class_index': 26}, 27: {'name': 'Ibizan_hound', 'display_name': 'Ibizan Hound', 'original_dir': 'n02091244-Ibizan_hound', 'image_count': 188, 'class_index': 27}, 28: {'name': 'bloodhound', 'display_name': 'Bloodhound', 'original_dir': 'n02088466-bloodhound', 'image_count': 187, 'class_index': 28}, 29: {'name': 'whippet', 'display_name': 'Whippet', 'original_dir': 'n02091134-whippet', 'image_count': 187, 'class_index': 29}, 30: {'name': 'Lhasa', 'display_name': 'Lhasa', 'original_dir': 'n02098413-Lhasa', 'image_count': 186, 'class_index': 30}, 31: {'name': 'Japanese_spaniel', 'display_name': 'Japanese Spaniel', 'original_dir': 'n02085782-Japanese_spaniel', 'image_count': 185, 'class_index': 31}, 32: {'name': 'Norwich_terrier', 'display_name': 'Norwich Terrier', 'original_dir': 'n02094258-Norwich_terrier', 'image_count': 185, 'class_index': 32}, 33: {'name': 'miniature_pinscher', 'display_name': 'Miniature Pinscher', 'original_dir': 'n02107312-miniature_pinscher', 'image_count': 184, 'class_index': 33}, 34: {'name': 'silky_terrier', 'display_name': 'Silky Terrier', 'original_dir': 'n02097658-silky_terrier', 'image_count': 183, 'class_index': 34}, 35: {'name': 'Italian_greyhound', 'display_name': 'Italian Greyhound', 'original_dir': 'n02091032-Italian_greyhound', 'image_count': 182, 'class_index': 35}, 36: {'name': 'Bedlington_terrier', 'display_name': 'Bedlington Terrier', 'original_dir': 'n02093647-Bedlington_terrier', 'image_count': 182, 'class_index': 36}, 37: {'name': 'Boston_bull', 'display_name': 'Boston Bull', 'original_dir': 'n02096585-Boston_bull', 'image_count': 182, 'class_index': 37}, 38: {'name': 'Pembroke', 'display_name': 'Pembroke', 'original_dir': 'n02113023-Pembroke', 'image_count': 181, 'class_index': 38}, 39: {'name': 'Dandie_Dinmont', 'display_name': 'Dandie Dinmont', 'original_dir': 'n02096437-Dandie_Dinmont', 'image_count': 180, 'class_index': 39}, 40: {'name': 'Kerry_blue_terrier', 'display_name': 'Kerry Blue Terrier', 'original_dir': 'n02093859-Kerry_blue_terrier', 'image_count': 179, 'class_index': 40}, 41: {'name': 'malamute', 'display_name': 'Malamute', 'original_dir': 'n02110063-malamute', 'image_count': 178, 'class_index': 41}, 42: {'name': 'basset', 'display_name': 'Basset', 'original_dir': 'n02088238-basset', 'image_count': 175, 'class_index': 42}, 43: {'name': 'toy_terrier', 'display_name': 'Toy Terrier', 'original_dir': 'n02087046-toy_terrier', 'image_count': 172, 'class_index': 43}, 44: {'name': 'Rhodesian_ridgeback', 'display_name': 'Rhodesian Ridgeback', 'original_dir': 'n02087394-Rhodesian_ridgeback', 'image_count': 172, 'class_index': 44}, 45: {'name': 'Border_terrier', 'display_name': 'Border Terrier', 'original_dir': 'n02093754-Border_terrier', 'image_count': 172, 'class_index': 45}, 46: {'name': 'Norfolk_terrier', 'display_name': 'Norfolk Terrier', 'original_dir': 'n02094114-Norfolk_terrier', 'image_count': 172, 'class_index': 46}, 47: {'name': 'bluetick', 'display_name': 'Bluetick', 'original_dir': 'n02088632-bluetick', 'image_count': 171, 'class_index': 47}, 48: {'name': 'Labrador_retriever', 'display_name': 'Labrador Retriever', 'original_dir': 'n02099712-Labrador_retriever', 'image_count': 171, 'class_index': 48}, 49: {'name': 'Saint_Bernard', 'display_name': 'Saint Bernard', 'original_dir': 'n02109525-Saint_Bernard', 'image_count': 170, 'class_index': 49}}}

# Quick mapping: breed name -> class index for label encoding
BREED_NAME_TO_INDEX = {
    "Maltese_dog": 0,
    "Afghan_hound": 1,
    "Scottish_deerhound": 2,
    "Pomeranian": 3,
    "Irish_wolfhound": 4,
    "Bernese_mountain_dog": 5,
    "Samoyed": 6,
    "Shih-Tzu": 7,
    "Great_Pyrenees": 8,
    "Leonberg": 9,
    "basenji": 10,
    "Tibetan_terrier": 11,
    "Sealyham_terrier": 12,
    "Airedale": 13,
    "EntleBucher": 14,
    "Saluki": 15,
    "pug": 16,
    "Lakeland_terrier": 17,
    "cairn": 18,
    "papillon": 19,
    "Norwegian_elkhound": 20,
    "Australian_terrier": 21,
    "chow": 22,
    "beagle": 23,
    "Newfoundland": 24,
    "Siberian_husky": 25,
    "Blenheim_spaniel": 26,
    "Ibizan_hound": 27,
    "bloodhound": 28,
    "whippet": 29,
    "Lhasa": 30,
    "Japanese_spaniel": 31,
    "Norwich_terrier": 32,
    "miniature_pinscher": 33,
    "silky_terrier": 34,
    "Italian_greyhound": 35,
    "Bedlington_terrier": 36,
    "Boston_bull": 37,
    "Pembroke": 38,
    "Dandie_Dinmont": 39,
    "Kerry_blue_terrier": 40,
    "malamute": 41,
    "basset": 42,
    "toy_terrier": 43,
    "Rhodesian_ridgeback": 44,
    "Border_terrier": 45,
    "Norfolk_terrier": 46,
    "bluetick": 47,
    "Labrador_retriever": 48,
    "Saint_Bernard": 49,
}

# Quick mapping: class index -> display name for human-readable output
BREED_INDEX_TO_DISPLAY = {
    0: "Maltese Dog",
    1: "Afghan Hound",
    2: "Scottish Deerhound",
    3: "Pomeranian",
    4: "Irish Wolfhound",
    5: "Bernese Mountain Dog",
    6: "Samoyed",
    7: "Shih-Tzu",
    8: "Great Pyrenees",
    9: "Leonberg",
    10: "Basenji",
    11: "Tibetan Terrier",
    12: "Sealyham Terrier",
    13: "Airedale",
    14: "Entlebucher",
    15: "Saluki",
    16: "Pug",
    17: "Lakeland Terrier",
    18: "Cairn",
    19: "Papillon",
    20: "Norwegian Elkhound",
    21: "Australian Terrier",
    22: "Chow",
    23: "Beagle",
    24: "Newfoundland",
    25: "Siberian Husky",
    26: "Blenheim Spaniel",
    27: "Ibizan Hound",
    28: "Bloodhound",
    29: "Whippet",
    30: "Lhasa",
    31: "Japanese Spaniel",
    32: "Norwich Terrier",
    33: "Miniature Pinscher",
    34: "Silky Terrier",
    35: "Italian Greyhound",
    36: "Bedlington Terrier",
    37: "Boston Bull",
    38: "Pembroke",
    39: "Dandie Dinmont",
    40: "Kerry Blue Terrier",
    41: "Malamute",
    42: "Basset",
    43: "Toy Terrier",
    44: "Rhodesian Ridgeback",
    45: "Border Terrier",
    46: "Norfolk Terrier",
    47: "Bluetick",
    48: "Labrador Retriever",
    49: "Saint Bernard",
}
