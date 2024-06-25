import json
import os
import string
from glob import glob
import numpy as np
from tqdm import tqdm
from utils import plot_image_with_caption


def patterns(l):
    return [f" {l}, ", f" ({l})"]  # create patterns for detecting sub-figure identifier: A, xxx. / (A) xxx. / xxx (A).


def which_pattern(splitter):
    if splitter.__contains__(","):
        return 0
    else:
        return 1


def get_letter(splitter):
    if which_pattern(splitter) == 0:
        return splitter[1]
    else:
        return splitter[2]


ALPHABETS = list(string.ascii_lowercase)
NUMBERS = [str(n) for n in range(1, 20)]
PATTERNS = sum([patterns(a) for a in ALPHABETS + NUMBERS], [])


def string_index(c, s):
    return [pos for pos, char in enumerate(s) if char == c]


def change_char_in_string(s, i, c):
    list_s = list(s)
    list_s[i] = c
    return ''.join(list_s)


def change_parentheses(s):
    # change (xxx) to [xxx], so that (x) can be detected as the sub-figure identifier
    res = s
    left = string_index('(', res)[::-1]  # process from right most (, parentheses may exist within another parentheses

    for i in left:
        first_right_loc = res[i + 1:].find(')')
        if first_right_loc > 1:  # avoid changing sub-figure identifier (a) to [a]
            res = change_char_in_string(res, i, '[')
            res = change_char_in_string(res, i + first_right_loc + 1, ']')
            left = string_index('(', res)[::-1]

    return res


def split(l, caption):
    """
    split caption based on the sub-figure identifier "l"
    step 1: split all sub-captions
    step 2: return the queried one
    """
    caption = ' ' + caption
    splitters = [s for s in PATTERNS if s in caption]  # detect existing sub-figure identifier in the caption
    if len(splitters) <= 1:
        return caption
    # normalize the style of all splitters to the style of the first one, e.g., [' a, ', ' (b)'] --> [' a, ', ' b,']
    style = [which_pattern(s) for s in splitters]
    if any(np.array(style[1:]) - style[0]):
        new_splitters = [patterns(get_letter(s))[style[0]] for s in splitters]
        for i in range(1, len(splitters)):
            caption = caption.replace(splitters[i], new_splitters[i])
        splitters = new_splitters

    res = []
    if any([s + '.' in caption for s in splitters]):  # for case where sub-figure identifier is at the end: xxx (A).
        for s in splitters:
            if s not in caption:
                res.append('')
            else:
                res.append(caption.split(s)[0])  # the sub-caption is before the identifier
                caption = caption.split(s)[1]  # crop the split sub-caption

        # sentences before the last one are regarded as super caption that is applicable for all sub-figures
        caption = ''.join(res[0].split('.')[:-1])
        res = [res[0], *[caption + '.' + r for r in res[1:]]]

    else:  # for cases where sub-figure identifier is at the beginning: A, xxx. | (A) xxx.
        for s in splitters[::-1]:
            if s not in caption:
                res.append('')
            else:
                res.append(caption.split(s)[1])  # the sub-caption is after the identifier
                caption = caption.split(s)[0]

        # super_caption: the rest of the caption
        res = [caption + r for r in res]
        res = res[::-1]

    try:
        # return the sub-caption corresponding to the query identifier "l", instead of all split sub-captions
        return res[[l in s for s in splitters].index(True)]
    except:
        return caption


def image_keyword_search(source_dataset_path, keyword_dict, retrieve_json, save_dir=None):
    """
    ARCH data structure:
    A bag denotes one figure from the book or publication, an instance of the bag denotes a subfigure of the figure
    In the following example, instances 0 and 1 belong to the bag 00.
    Note that all instances in the bag share the same caption without splitting

    book_Set:
    * `figure_id` - corresponds to the id of the bag
    * `letter` - corresponds to the id of the instance within the bag. `single` indicates that there is only a single instance within a bag.
    * `caption` - is the textual caption for that bag.
    * `uuid` - is the unique image identifier of that instance.

    example:
    {
      "0":{
        "figure_id":"00",
        "letter":"A",
        "caption":" A, Spindle cell variant of embryonal rhabdomyosarcoma is characterized by fascicles of eosinophilic spindle cells (B), some of which can show prominent paranuclear vacuolisation, as seen in leiomyosarcoma.",
        "uuid":"890e2e79-ab0a-4a2e-9d62-b0b6b3d43884"
      },
      "1":{
        "figure_id":"00",
        "letter":"B",
        "caption":" A, Spindle cell variant of embryonal rhabdomyosarcoma is characterized by fascicles of eosinophilic spindle cells (B), some of which can show prominent paranuclear vacuolisation, as seen in leiomyosarcoma.",
        "uuid":"f12c8088-05a5-41a6-80b8-aa4cfa461236"
      },
      "2":{
        "figure_id":"01",
        "letter":"Single",
        "caption":" In the anaplastic variant of embryonal rhabdomyosarcoma, the tumor cells have enlarged hyperchromatic and atypical nuclei. Note the presence of a tripolar mitotic figure.",
        "uuid":"9a77b172-74e8-4e64-878f-d26b7c27239f"
      },
    ...
    }

    pubmed_set (n=3309):
    * `caption` - is the textual caption for that bag.
    * `uuid` - is the unique image identifier of that instance.
    {
      "0":{
        "caption":"ER expression in tumor tissue. IHC staining, original",
        "uuid":"3f93c716-8fc9-42e9-bc29-bec52a51ab4b"
      },
      "1":{
        "caption":"Nuclear expression of TS (brown) in a colon carcinoma",
        "uuid":"9fcdf1e1-139c-4b63-bf1a-79d83c71f41a"
      },
      "2":{
        "caption":"Nuclear expression of E2F1 (brown) in a colon carcinoma. This is higher magnification of the upper portion of a core shown in an inset (lower left corner)",
        "uuid":"00f1ad7a-f4b0-4938-b874-089d40a123ce"
      },
      "3":{
        "caption":"Cytoplasmic immunoexpression of PD-L1 in oral squamous cell carcinomas with poorer prognosis (OSCCPP). Immunohistochemistry. Total magnification x100",
        "uuid":"9d3aef30-7c8b-4b78-9acf-ec523f952650"
      },
      "4":{
        "caption":"Nuclear and perinuclear immunoexpression of Foxp3 in oral squamous cell carcinomas with poorer prognosis (OSCCPP). Immunohistochemistry. Total magnification x100",
        "uuid":"b317d529-3626-49fc-9282-e4f28cf3d1cb"
      },
      "5":{
        "caption":"Cytoplasmic immunoexpression of PD-L1 in oral squamous cell carcinomas with better prognosis (OSCCBP). Immunohistochemistry. Total magnification x100",
        "uuid":"e9e99cb6-f795-4c5d-9d66-a8e81475b934"
      },
      "6":{
        "caption":"Nuclear and perinuclear immunoexpression of Foxp3 in oral squamous cell carcinomas with better prognosis (OSCCBP). Immunohistochemistry. Total magnification x100",
        "uuid":"c707e670-51d0-468a-b70d-ab01d7c68546"
      },
      ...
    }
    """
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(retrieve_json):
        with open(retrieve_json, 'r') as f:
            res = json.load(f)
    else:
        res = []
    already_retrieved = len(res)
    print(f"image_keyword_search on {source_dataset_path}... ({already_retrieved} existed pairs)")

    retrieved = []

    image_dir = os.path.join(source_dataset_path, "images")
    caption_dir = os.path.join(source_dataset_path, "captions.json")

    with open(caption_dir, 'r') as f:
        caption_dict = json.load(f)

    for id, v in tqdm(caption_dict.items()):
        found = True
        caption = v['caption'].casefold()
        caption = change_parentheses(caption)

        # split captions <-- in books_set, all sub-figures of a figure share the same caption without splitting
        if os.path.basename(source_dataset_path) == "books_set" and v['letter'] != 'Single':
            caption = split(v['letter'].casefold(), caption)
        v['caption'] = caption

        for key, value in keyword_dict.items():
            w, wo = [val.casefold() for val in value["with"]], [val.casefold() for val in value["without"]]

            if (w and not any([k in caption for k in w])) or any([k in caption for k in wo]):
                found = False
                break

        if found:
            retrieved.append(v)

    for id, r in tqdm(enumerate(retrieved)):
        caption, uuid = r['caption'], r['uuid']
        path = sum([glob(f"{image_dir}/{uuid}.{suffix}") for suffix in ['jpg', 'png', 'jpeg']], [])

        if not path:
            print(f"Image not found: {uuid}")
        else:
            res.append({
                "ID": id + already_retrieved,
                "filename": path[0],
                "caption": caption
            })

            if save_dir:
                plot_image_with_caption([res[-1]], save_dir)

    print(f"image_keyword_search on {source_dataset_path}... "
          f"found {len(res) - already_retrieved} new pairs "
          f"-> in total {len(res)} pairs")
    with open(retrieve_json, 'w') as f:
        json.dump(res, f)
