from typing import List, TypedDict


class Emotions(TypedDict):
    angry: float
    disgust: float
    fear: float
    happy: float
    sad: float
    surprise: float
    neutral: float


class Face(TypedDict):
    box: List[int]
    emotions: Emotions


def to_face(input) -> Face:
    return input


def to_dict(emo: Emotions) -> dict[str, float]:
    """
    Needed to satisfy typechecking.
    Gets an Emotion and converts it to a normal dict
    """
    d: dict[str, float] = {}
    d["angry"] = emo["angry"]
    d["disgust"] = emo["disgust"]
    d["fear"] = emo["fear"]
    d["happy"] = emo["happy"]
    d["sad"] = emo["sad"]
    d["surprise"] = emo["surprise"]
    d["neutral"] = emo["neutral"]
    for key in emo.keys():
        # needed because if there are more entries in the future,
        # we want to add them here aswell
        assert key in d
    return d


def to_faces(input: list) -> list[Face]:
    """
    Needed for better development experience.
    Converts the Output of the FER into a better Datatype
    """
    list = []
    for face in input:
        list.append(to_face(face))
    return list
