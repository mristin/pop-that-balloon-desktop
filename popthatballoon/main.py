"""Pop the balloons using your body pose estimation."""
import argparse
import collections
import concurrent.futures
import enum
import fractions
import importlib.resources
import os
import pathlib
import random
import sys
import time
from typing import List, Optional, Final, Mapping, Callable, MutableMapping, Tuple

import cv2
import numpy as np
import pygame
import tensorflow as tf
import tensorflow_hub as hub
from icontract import require

import popthatballoon
from popthatballoon.common import assert_never

assert popthatballoon.__doc__ == __doc__

PACKAGE_DIR = (
    pathlib.Path(str(importlib.resources.files(__package__)))
    if __package__ is not None
    else pathlib.Path(os.path.realpath(__file__)).parent
)


class Paths:
    """Wire the paths to media files."""

    def __init__(self) -> None:
        self.all = []  # type: List[str]

        @require(lambda path: not os.path.isabs(path))
        def prepend_package_dir_and_register(path: str) -> str:
            """Register the path with :py:attr:`all_paths`."""
            absolute = os.path.join(PACKAGE_DIR, path)
            self.all.append(absolute)
            return absolute

        self.balloon_idling_images = [
            prepend_package_dir_and_register(f"media/sprites/balloon_idling{i}.png")
            for i in range(3)
        ]
        self.balloon_popping_images = [
            prepend_package_dir_and_register(f"media/sprites/balloon_popping{i}.png")
            for i in range(3)
        ]
        self.hourglass_images = [
            prepend_package_dir_and_register(f"media/sprites/hourglass{i}.png")
            for i in range(5)
        ]

        self.font = prepend_package_dir_and_register("media/freesansbold.ttf")

        self.plop_sound = prepend_package_dir_and_register("media/sfx/plop.ogg")


def check_all_files_exist(paths: Paths) -> Optional[str]:
    """Check that all files exist, and return an error, if any."""
    for path in paths.all:
        if not os.path.exists(path):
            return f"The media file does not exist: {path}"

    return None


class KeypointLabel(enum.Enum):
    """Map keypoints names to the indices in the network output."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


#: Map indices of the network output to keypoint labels
KEYPOINT_INDEX_TO_LABEL = {literal.value: literal for literal in KeypointLabel}


class Keypoint:
    """Represent a single detection of a keypoint in an image."""

    x: float
    y: float
    confidence: float

    @require(lambda confidence: 0 <= confidence <= 1)
    def __init__(self, x: float, y: float, confidence: float) -> None:
        """
        Initialize with the given values.

        :param x: X-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param y: Y-coordinate in the image rescaled to [0, 1] x [0, 1]
        :param confidence: in the range [0,1] of the keypoint detection
        """
        self.x = x
        self.y = y
        self.confidence = confidence


class Detection:
    """Represent a detection of persons in an image."""

    #: Keypoints of the pose
    keypoints: Final[Mapping[KeypointLabel, Keypoint]]

    #: Score of the person detection.
    #:
    #: .. note::
    #:
    #:     This score is the score of the *person* detection, not of the individual
    #:     joints. For the score of the individual joints,
    #:     see :py:attr:`Keypoint.confidence`

    @require(lambda score: 0 <= score <= 1)
    def __init__(
        self,
        keypoints: Mapping[KeypointLabel, Keypoint],
        score: float,
    ) -> None:
        """Initialize with the given values."""
        self.keypoints = keypoints
        self.score = score


def load_detector() -> Callable[[cv2.Mat], List[Detection]]:
    """
    Load the model and return the function which you can readily use on images.

    :return: function to be applied on images
    """
    model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
    movenet = model.signatures["serving_default"]

    # If a detection has a score below this threshold, it will be ignored.
    detection_score_threshold = 0.2

    # If a keypoint has a confidence below this threshold, it will be ignored.
    keypoint_confidence_threshold = 0.2

    def apply_model(img: cv2.Mat) -> List[Detection]:
        # NOTE (mristin, 2022-12-23):
        # Vaguely based on:
        # * https://www.tensorflow.org/hub/tutorials/movenet,
        # * https://www.section.io/engineering-education/multi-person-pose-estimator-with-python/,
        # * https://analyticsindiamag.com/how-to-do-pose-estimation-with-movenet/ and
        # * https://github.com/geaxgx/openvino_movenet_multipose/blob/main/MovenetMPOpenvino.py

        # Both height and width need to be multiple of 32,
        # height to width ratio should resemble the original image, and
        # the larger side should be made to 256 pixels.
        #
        # Example: 720x1280 should be resized to 160x256.

        height, width, _ = img.shape

        input_size = 256

        if height > width:
            new_height = input_size
            # fmt: off
            new_width = int(
                (float(width) * float(new_height) / float(height)) // 32
            ) * 32
            # fmt: on
        else:
            new_width = input_size
            # fmt: off
            new_height = int(
                (float(height) * float(new_width) / float(width)) // 32
            ) * 32
            # fmt: on

        if new_height != height or new_width != width:
            resized = cv2.resize(img, (new_width, new_height))
        else:
            resized = img

        tf_input_img = tf.cast(
            tf.image.resize_with_pad(
                image=tf.expand_dims(resized, axis=0),
                target_height=new_height,
                target_width=new_width,
            ),
            dtype=tf.int32,
        )

        inference = movenet(tf_input_img)
        output_as_tensor = inference["output_0"]
        assert output_as_tensor.shape == (1, 6, 56)

        output = np.squeeze(output_as_tensor)
        assert output.shape == (6, 56)

        detections = []  # type: List[Detection]

        for i in range(6):
            kps = output[i][:51].reshape(17, -1)
            bbox = output[i][51:55].reshape(2, 2)
            score = output[i][55]

            if score < detection_score_threshold:
                continue

            assert kps.shape == (17, 3)
            assert bbox.shape == (2, 2)

            kps_xy = kps[:, [1, 0]]
            kps_confidence = kps[:, 2]

            assert kps_xy.shape == (17, 2)
            assert kps_confidence.shape == (17,)

            keypoints = (
                collections.OrderedDict()
            )  # type: MutableMapping[KeypointLabel, Keypoint]

            for i in range(17):
                label = KEYPOINT_INDEX_TO_LABEL[i]
                kp_x, kp_y = kps_xy[i, :]
                kp_confidence = kps_confidence[i]

                if kp_confidence < keypoint_confidence_threshold:
                    continue

                assert label not in keypoints
                keypoints[label] = Keypoint(kp_x, kp_y, kp_confidence)

            detection = Detection(keypoints, score)

            detections.append(detection)

        return detections

    return apply_model


class BalloonAction(enum.Enum):
    """Capture the possible actions of a balloon."""

    IDLING = 0
    POPPING = 1


class BalloonState:
    """Capture the state of a balloon."""

    #: Top-left corner
    position_xy: Tuple[int, int]

    action: BalloonAction

    #: Seconds since epoch when the state started
    start: float

    #: Seconds since epoch when the state is to end; if not set, endless until change
    end: Optional[float]

    def __init__(
        self,
        position_xy: Tuple[int, int],
        action: BalloonAction,
        start: float,
        end: Optional[float],
    ) -> None:
        """Initialize with the given values."""
        self.position_xy = position_xy
        self.action = action
        self.start = start
        self.end = end


class State:
    """Represent the state of the game."""

    left_hand_xy: Optional[Tuple[int, int]]
    right_hand_xy: Optional[Tuple[int, int]]

    #: Seconds since epoch
    next_balloon: float

    #: Balloon states
    balloons: List[BalloonState]

    game_over: bool

    quit: bool

    score: int

    #: In seconds since epoch
    game_start: float

    #: Planned game end, seconds since epoch
    game_end: float

    def __init__(self, game_start: float) -> None:
        self.left_hand_xy = None
        self.right_hand_xy = None

        self.next_balloon = game_start + 3
        self.balloons = []
        self.game_over = False
        self.quit = False
        self.score = 0

        self.game_start = game_start
        self.game_end = game_start + 60


BALLOON_SIZE = (50, 100)
WEAPON_SIZE = (10, 10)


@require(lambda xmin_a, xmax_a: xmin_a <= xmax_a)
@require(lambda ymin_a, ymax_a: ymin_a <= ymax_a)
@require(lambda xmin_b, xmax_b: xmin_b <= xmax_b)
@require(lambda ymin_b, ymax_b: ymin_b <= ymax_b)
def intersect(
    xmin_a: int,
    ymin_a: int,
    xmax_a: int,
    ymax_a: int,
    xmin_b: int,
    ymin_b: int,
    xmax_b: int,
    ymax_b: int,
) -> bool:
    """Return true if the two bounding boxes intersect."""
    return (xmin_a <= xmax_b and xmax_a >= xmin_b) and (
        ymin_a <= ymax_b and ymax_a >= ymin_b
    )


def bounding_boxes_of_weapons(state: State) -> List[Tuple[int, int, int, int]]:
    """
    Compute the bounding boxes of all the weapons in the image.

    Return a list of (xmin, ymin, xmax, ymax) where both min and max are *inclusive*.
    """
    weapons = []  # type: List[Tuple[int, int, int, int]]
    for kp_xy in [state.left_hand_xy, state.right_hand_xy]:
        if kp_xy is None:
            continue

        weapons.append(
            (
                kp_xy[0] - WEAPON_SIZE[0] // 2,
                kp_xy[1] - WEAPON_SIZE[1] // 2,
                kp_xy[0] + WEAPON_SIZE[0] // 2 - 1,
                kp_xy[1] + WEAPON_SIZE[1] // 2 - 1,
            )
        )

    return weapons


SOUND_CACHE = dict()  # type: MutableMapping[str, pygame.mixer.Sound]


def play_sound(path: str) -> float:
    """Start playing the sound and returns its length."""
    sound = SOUND_CACHE.get(path, None)
    if sound is None:
        sound = pygame.mixer.Sound(path)
        SOUND_CACHE[path] = sound

    sound.play()
    return sound.get_length()


def tick(state: State, frame_size: Tuple[int, int], paths: Paths) -> None:
    """Update state on a loop iteration and perform side effects."""
    now = time.time()

    if now > state.game_end:
        state.game_over = True
        return

    if now > state.next_balloon:
        margin_x = 0.1 * frame_size[0]
        margin_y = 0.1 * frame_size[1]

        max_space_x = frame_size[0] - BALLOON_SIZE[0] - margin_x
        max_space_y = frame_size[1] - BALLOON_SIZE[1] - margin_y

        state.balloons.append(
            BalloonState(
                position_xy=(
                    int(margin_x + max_space_x * random.random()),
                    int(margin_y + max_space_y * random.random()),
                ),
                action=BalloonAction.IDLING,
                start=now,
                end=None,
            )
        )

        state.next_balloon = now + random.random() * 2

    new_balloons = []  # type: List[BalloonState]
    for balloon_state in state.balloons:
        if balloon_state.action is BalloonAction.POPPING:
            assert balloon_state.end is not None
            if now > balloon_state.end:
                continue

        new_balloons.append(balloon_state)
    state.balloons = new_balloons

    weapons = bounding_boxes_of_weapons(state)
    for balloon_state in state.balloons:
        if balloon_state.action is BalloonAction.POPPING:
            continue

        for weapon in weapons:
            if intersect(
                weapon[0],
                weapon[1],
                weapon[2],
                weapon[3],
                balloon_state.position_xy[0],
                balloon_state.position_xy[1],
                balloon_state.position_xy[0] + BALLOON_SIZE[0] - 1,
                balloon_state.position_xy[1] + BALLOON_SIZE[1] - 1,
            ):
                balloon_state.start = now
                balloon_state.end = now + 0.25
                balloon_state.action = BalloonAction.POPPING
                state.score += 1

                play_sound(paths.plop_sound)
                break


IMAGE_CACHE = dict()  # type: MutableMapping[str, cv2.Mat]


def load_image_or_retrieve_from_cache(path: str) -> cv2.Mat:
    """Retrieve the image from the cache or load it."""
    image = IMAGE_CACHE.get(path, None)
    if image is not None:
        return image

    image = pygame.image.load(path).convert_alpha()
    IMAGE_CACHE[path] = image
    return image


def render_game_over(
    state: State, surface: pygame.surface.Surface, paths: Paths
) -> None:
    """Render the "game over" dialogue."""
    oneph = max(1, int(0.01 * surface.get_height()))
    onepw = max(1, int(0.01 * surface.get_height()))

    surface.fill((0, 0, 0))

    font_large = pygame.font.Font(paths.font, 5 * oneph)
    game_over = font_large.render("Game Over", True, (255, 255, 255))
    game_over_xy = (10 * onepw, 10 * oneph)
    surface.blit(game_over, game_over_xy)

    score = font_large.render(f"Score: {state.score}", True, (255, 255, 255))
    score_xy = (game_over_xy[0], game_over_xy[1] + game_over.get_height() + oneph)
    surface.blit(score, score_xy)

    font_small = pygame.font.Font(paths.font, 2 * oneph)
    escape = font_small.render('Press ESC or "q" to quit', True, (255, 255, 255))
    escape_xy = (score_xy[0], surface.get_height() - escape.get_height() - 2 * oneph)
    surface.blit(escape, escape_xy)

    font_medium = pygame.font.Font(paths.font, 4 * oneph)
    repeat = font_medium.render('Press "r" to restart', True, (0, 0, 255))
    repeat_xy = (escape_xy[0], escape_xy[1] - repeat.get_height() - oneph)
    surface.blit(repeat, repeat_xy)


def render_loading(surface: pygame.surface.Surface, paths: Paths) -> None:
    """Render the "game over" dialogue."""
    surface.fill((0, 0, 0))

    oneph = max(1, int(0.01 * surface.get_height()))
    onepw = max(1, int(0.01 * surface.get_height()))

    font_large = pygame.font.Font(paths.font, 5 * oneph)
    loading = font_large.render("Loading...", True, (255, 255, 255))
    loading_xy = (3 * onepw, 10 * oneph)
    surface.blit(loading, loading_xy)

    font_small = pygame.font.Font(paths.font, 2 * oneph)
    escape = font_small.render('Press ESC or "q" to quit', True, (255, 255, 255))
    escape_xy = (3 * onepw, surface.get_height() - escape.get_height() - 2 * oneph)
    surface.blit(escape, escape_xy)


def render_game(
    state: State,
    surface: pygame.surface.Surface,
    frame: pygame.surface.Surface,
    paths: Paths,
) -> None:
    """Render the game on the screen."""
    # Draw everything on the frame, then rescale the frame.
    #
    # This is a much easier approach than scaling everything to percentages of
    # the screen size as we can simply use absolute pixel values for positioning.
    scene = frame.copy()

    font_large = pygame.font.Font(paths.font, 32)
    score = font_large.render(f"Score: {state.score}", True, (0, 0, 255))
    score_xy = (5, 5)
    scene.blit(score, score_xy)

    weapons = bounding_boxes_of_weapons(state)
    for xmin, ymin, xmax, ymax in weapons:
        pygame.draw.rect(scene, (255, 0, 0), (xmin, ymin, xmax - xmin, ymax - ymin), 3)

    now = time.time()
    for balloon_state in state.balloons:
        duration = now - balloon_state.start

        sprite = None  # type: Optional[pygame.surface.Surface]
        if balloon_state.action is BalloonAction.IDLING:
            sprite_index = int(duration) % len(paths.balloon_idling_images)
            sprite = load_image_or_retrieve_from_cache(
                paths.balloon_idling_images[sprite_index]
            )

        elif balloon_state.action is BalloonAction.POPPING:
            assert balloon_state.end is not None

            time_fraction = duration / (balloon_state.end - balloon_state.start)
            sprite_index = min(
                len(paths.balloon_popping_images) - 1,
                int(time_fraction * len(paths.balloon_popping_images)),
            )
            sprite = load_image_or_retrieve_from_cache(
                paths.balloon_popping_images[sprite_index]
            )

        else:
            assert_never(balloon_state.action)

        assert sprite is not None

        scene.blit(sprite, balloon_state.position_xy)

    game_time_fraction = (now - state.game_start) / (state.game_end - state.game_start)

    hourglass_step = min(
        int(game_time_fraction * len(paths.hourglass_images)),
        len(paths.hourglass_images) - 1,
    )

    hourglass_path = paths.hourglass_images[hourglass_step]

    hourglass = load_image_or_retrieve_from_cache(hourglass_path)
    scene.blit(
        hourglass,
        (
            scene.get_width() - hourglass.get_width() - 3,
            3,
        ),
    )

    font_small = pygame.font.Font(paths.font, 16)
    escape = font_small.render('Press ESC or "q" to quit', True, (0, 0, 255))
    escape_xy = (4, scene.get_height() - escape.get_height() - 2)
    scene.blit(escape, escape_xy)

    # Now draw the scene on the screen.

    surface.fill((0, 0, 0))

    surface_aspect_ratio = fractions.Fraction(surface.get_width(), surface.get_height())
    scene_aspect_ratio = fractions.Fraction(scene.get_width(), scene.get_height())

    if scene_aspect_ratio < surface_aspect_ratio:
        new_scene_height = surface.get_height()
        new_scene_width = scene.get_width() * (new_scene_height / scene.get_height())

        scene = pygame.transform.scale(scene, (new_scene_width, new_scene_height))

        margin = int((surface.get_width() - scene.get_width()) / 2)

        surface.blit(scene, (margin, 0))

    elif scene_aspect_ratio == surface_aspect_ratio:
        new_scene_width = surface.get_width()
        new_scene_height = scene.get_height()

        scene = pygame.transform.scale(scene, (new_scene_width, new_scene_height))

        surface.blit(scene, (0, 0))
    else:
        new_scene_width = surface.get_width()
        new_scene_height = int(
            scene.get_height() * (new_scene_width / scene.get_width())
        )

        scene = pygame.transform.scale(scene, (new_scene_width, new_scene_height))

        margin = int((surface.get_height() - scene.get_height()) / 2)

        surface.blit(scene, (0, margin))


def render(
    state: State,
    surface: pygame.surface.Surface,
    frame: pygame.surface.Surface,
    paths: Paths,
) -> None:
    """Render the game on the screen."""
    if state.game_over:
        render_game_over(state, surface, paths)
    else:
        render_game(state, surface, frame, paths)


def render_quitting(surface: pygame.surface.Surface, paths: Paths) -> None:
    """Render the "Quitting..." dialogue."""
    oneph = max(1, int(0.01 * surface.get_height()))
    onepw = max(1, int(0.01 * surface.get_height()))

    surface.fill((0, 0, 0))

    font_large = pygame.font.Font(paths.font, 5 * oneph)
    quitting = font_large.render("Quitting the game...", True, (255, 255, 255))

    quitting_xy = (10 * onepw, 10 * oneph)
    surface.blit(quitting, quitting_xy)


def cvmat_to_surface(image: cv2.Mat) -> pygame.surface.Surface:
    """Convert from OpenCV to pygame."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    return pygame.image.frombuffer(image_rgb.tobytes(), (width, height), "RGB")


def main(prog: str) -> int:
    """
    Execute the main routine.

    :param prog: name of the program to be displayed in the help
    :return: exit code
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument(
        "--version", help="show the current version and exit", action="store_true"
    )

    # NOTE (mristin, 2022-12-23):
    # The module ``argparse`` is not flexible enough to understand special options such
    # as ``--version`` so we manually hard-wire.
    if "--version" in sys.argv and "--help" not in sys.argv:
        print(popthatballoon.__version__)
        return 0

    parser.parse_args()

    paths = Paths()
    error = check_all_files_exist(paths)
    if error is not None:
        print(error, file=sys.stderr)
        return 1

    pygame.init()

    pygame.mixer.pre_init()
    pygame.mixer.init()

    pygame.display.set_caption("Pop-that-balloon")
    surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

    render_loading(surface, paths)
    pygame.display.flip()

    cap = None  # type: Optional[cv2.VideoCapture]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_detector = executor.submit(load_detector)

            while not future_detector.done() and not future_detector.cancelled():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        print("Quitting during the loading...")
                        render_quitting(surface, paths)
                        pygame.display.flip()
                        future_detector.cancel()
                        return 0
                    else:
                        # Ignore the event
                        pass

                # A display flip is necessary here; otherwise the pygame.event.get()
                # blocks.
                pygame.display.flip()

            detector = future_detector.result()

        state = State(game_start=time.time())

        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not state.quit:
            reading_ok, frame = cap.read()
            if not reading_ok:
                break

            # Flip so that it is easier to understand the image
            frame = cv2.flip(frame, 1)

            frame_h, frame_w, _ = frame.shape
            detections = detector(frame)
            if len(detections) > 0:
                detection = detections[0]

                kp_left_hand = detection.keypoints.get(KeypointLabel.LEFT_WRIST, None)
                if kp_left_hand is not None:
                    state.left_hand_xy = (
                        int(kp_left_hand.x * frame_w),
                        int(kp_left_hand.y * frame_h),
                    )
                else:
                    state.left_hand_xy = None

                kp_right_hand = detection.keypoints.get(KeypointLabel.RIGHT_WRIST, None)
                if kp_right_hand is not None:
                    state.right_hand_xy = (
                        int(kp_right_hand.x * frame_w),
                        int(kp_right_hand.y * frame_h),
                    )
                else:
                    state.right_hand_xy = None

            tick(state, (frame_w, frame_h), paths)

            frame_pygame = cvmat_to_surface(frame)
            render(state, surface, frame_pygame, paths)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state.quit = True
                    state.game_over = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    state = State(game_start=time.time())
                elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    state.quit = True
                    state.game_over = True
                else:
                    pass

    except Exception as exception:
        exc_type, _, exc_tb = sys.exc_info()
        assert (
            exc_tb is not None
        ), "Expected a traceback as we do not do anything fancy here"

        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        exc_type_name = getattr(exc_type, "__name__", None)
        if exc_type_name is None:
            exc_type_name = str(exc_type)

        print(
            f"Failed to process the video: "
            f"{exc_type_name} at {fname}:{exc_tb.tb_lineno} {exception}",
            file=sys.stderr,
        )
        return 1

    finally:
        if cap is not None:
            cap.release()

        print("Quitting...")
        render_quitting(surface, paths)
        pygame.display.flip()

        pygame.quit()

    return 0


def entry_point() -> int:
    """Provide an entry point for a console script."""
    return main(prog="pop-that-balloon")


if __name__ == "__main__":
    sys.exit(main(prog="pop-that-balloon"))
