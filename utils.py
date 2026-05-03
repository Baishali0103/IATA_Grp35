import os

size2id = {"small": 0, "big": 1}
id2size = {v: k for k, v in size2id.items()}

color2id = {"red": 0, "blue": 1, "green": 2, "yellow": 3}
id2color = {v: k for k, v in color2id.items()}

shape2id = {"circle": 0, "square": 1, "triangle": 2}
id2shape = {v: k for k, v in shape2id.items()}

relation2id = {"above": 0, "below": 1, "left of": 2, "right of": 3}
id2relation = {v: k for k, v in relation2id.items()}


def build_sentence_from_labels(pred):
    return (
        f"a {id2size[pred['obj1_size']]} {id2color[pred['obj1_color']]} {id2shape[pred['obj1_shape']]} "
        f"is {id2relation[pred['relation']]} "
        f"a {id2size[pred['obj2_size']]} {id2color[pred['obj2_color']]} {id2shape[pred['obj2_shape']]}"
    )


def get_image_path(image_dir, sample_id):
    """
    Match filenames like:
    sample_00.png
    sample_09.png
    sample_10.png
    sample_999.png
    """
    return os.path.join(image_dir, f"sample_{int(sample_id):02d}.png")