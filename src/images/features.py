from collections.abc import Callable
from typing import Optional
import numpy as np
import cv2
from skimage.feature import hog
import src.utils as utils
import matplotlib.pyplot as plt


def angle(p1: np.ndarray, p2: np.ndarray) -> float:
    diff = p1 - p2
    return np.arctan2(diff[1], diff[0])  # * 180 / np.pi


def angle_3_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cosine_angle)


def rotate_points(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    local_points = points.copy().reshape((-1, 2))
    # rotated = np.tensordot(local_points, rotation_matrix, axes=([1], [1]))
    local_points = np.hstack(
        (local_points, np.ones((local_points.shape[0], 1))))

    new_points = []
    for point in local_points:
        new_points.append(rotation_matrix.dot(point))

    return np.concatenate(new_points, axis=0)


def slope(p1: np.ndarray, p2: np.ndarray) -> float:
    diff = p1 - p2
    return abs(diff[1] / diff[0])


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum(np.square(p1 - p2)))


def perimeter(points: list[np.ndarray]) -> float:
    zipped = zip(points[:-1], points[1:])

    d = 0
    for pair in zipped:
        d += distance(pair[0], pair[1])

    return d + distance(points[-1], points[0])


def get_point(points, index):
    return points[index*2:index*2+2]


def dist_to_line(line_p1: np.ndarray, line_p2: np.ndarray, p3: np.ndarray) -> float:
    return np.abs(np.cross(line_p2-line_p1, p3-line_p1))/np.linalg.norm(line_p2-line_p1)*2


def poly_area(points: list[np.ndarray]):
    """
    Apply the Shoelace formula to calculate the area between points.
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def line_intersection(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    a = np.stack([a1, a2])
    b = np.stack([b1, b2])

    dets_a = a1 - a2
    dets_b = b1 - b2
    D = np.linalg.det(np.stack([dets_a, dets_b]))

    det_a = np.linalg.det(a)
    det_b = np.linalg.det(b)

    x = np.linalg.det(np.array([[det_a, dets_a[0]], [det_b, dets_b[0]]]))
    y = np.linalg.det(np.array([[det_a, dets_a[1]], [det_b, dets_b[1]]]))

    return np.array([x, y]) / D


class FacialFeaturesExtractor:
    geometric_features_num: int

    def calculate(self, image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        pass

    def draw(self, image: np.ndarray, points: np.ndarray) -> None:
        pass


class DominguezFeaturesExtractor(FacialFeaturesExtractor):
    """
    Calculate features from the article of Parra-Dominguez et al. By default points' indices are taken from
    the WFLW dataset annotation.
    """

    def __init__(self, points_mapper: Optional[Callable[[int], int]] = None, reference_points: tuple[int, int] = (0, 32)):
        """
        The points_mapper function is responsible for mapping indices of the WFLW annotations to other
        annotation space of your choice. It must be compatible with the array of points given to the
        calculate function.
        """
        super().__init__()
        self.points_indices = list(range(98))
        self.points_mapper = points_mapper if points_mapper is not None else lambda i: i
        self.P = {}  # Dict of points extracted from the numpy array
        self.geometric_features_num = 29
        self.reference_points = reference_points

    def draw(self, image: np.ndarray, points: np.ndarray) -> None:
        raise Exception('Not implemented')

    def calculate(self, image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        self._prepare_points(points)
        an = self._get_tilt_angle()
        points = self._rotate_points(points, an)
        image = self._rotate_image(image, an)
        self._prepare_points(points)
        features = self._calc_features(image, points)
        return image, points, an, features

    def _get_tilt_angle(self) -> float:
        # Rotate the face according to the relatively stable points
        p1 = self.P[str(self.reference_points[0])]
        p2 = self.P[str(self.reference_points[1])]
        return angle(p1, p2)

    def _rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        angle = angle * 180 / np.pi - 180
        p_rot_mat = cv2.getRotationMatrix2D(
            (0.5, 0.5), angle, scale=1)
        points = rotate_points(points, p_rot_mat)
        return points

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        return image.copy()

    def _prepare_points(self, points: np.ndarray):
        points_indices = self.points_indices
        P = {str(i): get_point(points, self.points_mapper(i))
             for i in points_indices}
        self.P = P

    def _calc_features(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        P = self.P
        # Notation used here is the same as in the article
        features = {}
        features['f0'] = abs(angle(P['33'], P['46']))
        features['f1'] = abs(angle(P['35'], P['44']))
        features['f2'] = abs(angle(P['37'], P['42']))

        # Height average of the 5 points within each eyebrow
        L_points = [33, 34, 35, 36, 37]
        L = np.average(np.array([P[str(i)][1] for i in L_points]))
        M_points = [42, 43, 44, 45, 46]
        M = np.average(np.array([P[str(i)][1] for i in M_points]))
        features['f3'] = max(L/M, M/L)

        features['f4'] = slope(P['33'], P['46'])
        features['f5'] = slope(P['35'], P['44'])
        features['f6'] = slope(P['37'], P['42'])
        features['f7'] = abs(angle(P['60'], P['72']))

        # Width of the eye (left and right)
        Bl = abs(P['60'][0] - P['64'][0])
        Br = abs(P['68'][0] - P['72'][0])
        features['f8'] = max(Bl/Br, Br/Bl)

        D = abs(P['0'][0] - P['60'][0])
        E = abs(P['72'][0] - P['32'][0])
        features['f9'] = max(D/E, E/D)

        H = distance(P['60'], P['55'])
        I = distance(P['72'], P['59'])
        features['f10'] = max(H/I, I/H)

        Nl = distance(P['61'], P['67'])
        Nr = distance(P['63'], P['65'])
        N = (Nl + Nr) / 2
        Ol = distance(P['69'], P['75'])
        Or = distance(P['71'], P['73'])
        O = (Ol + Or) / 2
        features['f11'] = max(N/O, O/N)
        features['f12'] = max(Nl/Or, Or/Nl)
        features['f13'] = max(Nr/Ol, Ol/Nr)
        features['f14'] = abs(angle(P['76'], P['82']))

        F = distance(P['60'], P['85'])
        G = distance(P['72'], P['85'])
        features['f15'] = max(F/G, G/F)

        Pl = distance(P['77'], P['87'])
        Ql = distance(P['81'], P['83'])
        features['f16'] = max(Pl/Ql, Ql/Pl)

        Pu = distance(P['78'], P['86'])
        Qu = distance(P['80'], P['84'])
        features['f17'] = max(Pu/Qu, Qu/Pu)

        Vl = distance(P['76'], P['85'])
        Vr = distance(P['82'], P['85'])
        A = distance(P['0'], P['32'])
        features['f18'] = max(Vl/A, Vr/A)

        W = abs(P['76'][0] - P['82'][0])
        Wl_points = [76, 77, 78, 79, 85, 86, 87]
        Wl = perimeter([P[str(i)] for i in Wl_points])
        Wr_points = [79, 80, 81, 82, 83, 84, 85]
        Wr = perimeter([P[str(i)] for i in Wr_points])
        features['f19'] = max(Pl/W, Ql/W)
        features['f20'] = max(Pu/W, Qu/W)
        features['f21'] = max(Wl/W, Wr/W)
        features['f22'] = abs(angle(P['55'], P['59']))
        features['f23'] = abs(angle(P['54'], P['85']))

        J = distance(P['55'], P['85'])
        K = distance(P['59'], P['85'])
        features['f24'] = max(J/K, K/J)

        T = distance(P['34'], P['85'])
        U = distance(P['45'], P['85'])
        features['f25'] = max(T/A, U/A)

        R = distance(P['36'], P['85'])
        S = distance(P['43'], P['85'])
        features['f26'] = max(R/A, S/A)

        C = distance(P['85'], P['16'])
        X = distance(P['54'], P['79'])
        features['f27'] = C/A
        features['f28'] = X/A

        res = np.array(list(features.values()))
        assert len(res) == self.geometric_features_num, "Wrong number of features"
        return res


class KarolewskiFeaturesExtractor(DominguezFeaturesExtractor):
    def __init__(self, points_mapper: Optional[Callable[[int], int]] = None, reference_points: tuple[int, int] = (64, 68)):
        super().__init__(points_mapper, reference_points)
        # Less points are required
        self.points_indices = list(range(98))
        self.geometric_features_num = 21

    # def _get_tilt_angle(self) -> float:
    #     # Rotate the face according to the relatively stable points (64 and 68)
    #     return angle(self.P['0'], self.P['32'])
    #     # return np.pi

    def draw(self, image: np.ndarray, points: np.ndarray, version: str = 'areas') -> None:
        if version == 'areas':
            image = self.draw_areas(image, points)
        elif version == 'center_points':
            image = self.draw_center_points(image, points)
        elif version == 'lines':
            image = self.draw_lines(image, points)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        f, a = plt.subplots(1, 1, figsize=(12, 12))
        a.axis('off')
        a.imshow(rgb_img)

    def draw_lines(self, image: np.ndarray, points: np.ndarray) -> None:
        _, _, an, features = self.calculate(image, points)
        P = self.P
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        color3 = (255, 0, 0)
        # Calculate points on the line that divides the face in half
        middle_point: np.ndarray = (P['68'] - P['64']) / 2
        perp_point = middle_point.copy()
        perp_point[0], perp_point[1] = -middle_point[1], middle_point[0]
        middle_point += P['64']
        perp_point += middle_point
        thickness = 2

        utils.draw_line(image, P['60'], P['16'], color3, 2)
        utils.draw_line(image, P['72'], P['16'], color3, 2)
        middle_point_0_32: np.ndarray = (P['32'] - P['0']) / 2
        middle_point_0_32 += P['0']

        image = utils.draw_line(
            image, P['10'], middle_point, color3, thickness)
        image = utils.draw_line(
            image, P['22'], middle_point, color3, thickness)
        image = utils.draw_line(image, P['10'], P['57'], color3, thickness)
        image = utils.draw_line(image, P['22'], P['57'], color3, thickness)

        # image = utils.draw_line(image, P['60'], P['76'], color3, thickness)
        # image = utils.draw_line(image, P['82'], P['72'], color3, thickness)
        # image = utils.draw_line(image, P['64'], P['76'], color3, thickness)
        # image = utils.draw_line(image, P['68'], P['82'], color3, thickness)

        utils.draw_point(image, P['10'], 5, color1, -1)
        utils.draw_point(image, P['22'], 5, color1, -1)
        utils.draw_point(image, middle_point, 5, color1, -1)
        utils.draw_point(image, P['57'], 5, color1, -1)

        points_to_draw = [60, 72, 16, 10, 22, 57]  # , 76, 82, 64, 68]
        points_to_draw = [P[str(i)] for i in points_to_draw]

        for point in points_to_draw:
            utils.draw_point(image, point, 4, color1, -1)

        return image

    def draw_center_points(self, image: np.ndarray, points: np.ndarray) -> None:
        image, points, an, features = self.calculate(image, points)
        P = self.P
        # Calculate points on the line that divides the face in half
        middle_point: np.ndarray = (P['68'] - P['64']) / 2
        perp_point = middle_point.copy()
        perp_point[0], perp_point[1] = -middle_point[1], middle_point[0]
        middle_point += P['64']
        perp_point += middle_point
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        color3 = (255, 0, 0)

        utils.draw_line_through_points(image, middle_point, perp_point)
        utils.draw_line(image, P['64'], P['68'], color3, 2)

        points_to_draw = [51, 57, 79, 85, 16]
        points_to_draw = [P[str(i)] for i in points_to_draw]

        for point in points_to_draw:
            point_on_line = middle_point.copy()
            point_on_line[0], point_on_line[1] = middle_point[0], point[1]
            utils.draw_line(image, point, point_on_line, color1, 2)

        points_to_draw += [P['64'], P['68']]
        for point in points_to_draw:
            utils.draw_point(image, point, 4, color1, -1)

        return image

    def draw_areas(self, image: np.ndarray, points: np.ndarray) -> None:
        image, points, an, features = self.calculate(image, points)
        P = self.P
        # Calculate points on the line that divides the face in half
        middle_point: np.ndarray = (P['68'] - P['64']) / 2
        perp_point = middle_point.copy()
        perp_point[0], perp_point[1] = -middle_point[1], middle_point[0]
        middle_point += P['64']
        perp_point += middle_point

        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        thickness = 1
        # points_to_draw = [51, 54, 57, 79, 85, 16]
        points_to_draw = [64, 68]
        points_to_draw = [P[str(i)] for i in points_to_draw]
        jowl_top_point = line_intersection(
            middle_point, perp_point, P['0'], P['32'])
        jowl_bottom_point = line_intersection(
            middle_point, perp_point, P['5'], P['27'])
        jaw_middle_point = line_intersection(
            middle_point, perp_point, P['15'], P['17'])
        mouth_top_point = line_intersection(
            middle_point, perp_point, P['78'], P['80'])
        mouth_bottom_point = line_intersection(
            middle_point, perp_point, P['86'], P['84'])
        intersection_points = [jowl_bottom_point, jowl_top_point,
                               jaw_middle_point, mouth_bottom_point, mouth_top_point]

        # image = cv2.line(image, p1, p2, color, thickness)
        points_to_draw += intersection_points
        overlay = image.copy()

        lines_to_draw = [P[str(i)] for i in range(0, 6)] + \
            [P[str(i)] for i in range(27, 33)]
        utils.draw_polygon(overlay, np.array(lines_to_draw), color=(255, 0, 0))

        lines_to_draw = [P[str(i)] for i in range(5, 28)]
        utils.draw_polygon(overlay, np.array(lines_to_draw), color=(0, 255, 0))

        lines_to_draw = [P[str(i)] for i in range(76, 88)]
        utils.draw_polygon(overlay, np.array(lines_to_draw), color=(0, 0, 255))

        lines_to_draw = [P[str(i)] for i in range(60, 68)]
        utils.draw_polygon(overlay, np.array(
            lines_to_draw), color=(0, 255, 255))

        lines_to_draw = [P[str(i)] for i in range(68, 75)]
        utils.draw_polygon(overlay, np.array(
            lines_to_draw), color=(0, 255, 255))

        image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)

        # for p1, p2 in zip(lines_to_draw, lines_to_draw[1:]):
        #     utils.draw_line(image, p1, p2, (255, 0, 0), 2)

        # lines_to_draw = [P[str(i)] for i in range(76, 88)]
        # for p1, p2 in zip(lines_to_draw, lines_to_draw[1:] + [P['76']]):
        #     utils.draw_line(image, p1, p2, (255, 0, 0), 2)

        # lines_to_draw = [P[str(i)] for i in range(60, 68)]
        # for p1, p2 in zip(lines_to_draw, lines_to_draw[1:] + lines_to_draw[:1]):
        #     utils.draw_line(image, p1, p2, (255, 0, 0), 2)

        # lines_to_draw = [P[str(i)] for i in range(68, 75)]
        # for p1, p2 in zip(lines_to_draw, lines_to_draw[1:] + lines_to_draw[:1]):
        #     utils.draw_line(image, p1, p2, (255, 0, 0), 2)

        utils.draw_line_through_points(image, middle_point, perp_point)
        utils.draw_line(image, P['64'], P['68'], (255, 0, 0), 2)
        # utils.draw_line(image, P['5'], P['27'], (255, 0, 0), 2)

        for point in points_to_draw:
            utils.draw_point(image, point,
                             5, (0, 0, 255), -1)

        Ul = poly_area([jowl_top_point] + [P[str(i)]
                       for i in range(0, 6)] + [jowl_bottom_point])
        Ur = poly_area([jowl_bottom_point] + [P[str(i)]
                       for i in range(27, 33)] + [jowl_top_point])
        Ul_p, Ur_p = Ul / (Ul + Ur), Ur / (Ul + Ur)

        utils.draw_text(image, f'{Ul_p * 100:.2f}%', jowl_top_point - np.array(
            [0.25, -0.1]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4)
        utils.draw_text(image, f'{Ur_p * 100:.2f}%', jowl_top_point - np.array(
            [-0.15, -0.1]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4)

        # Jaw areas
        Bl = poly_area([jowl_bottom_point] + [P[str(i)]
                       for i in range(5, 16)] + [jaw_middle_point])
        Br = poly_area([jaw_middle_point] + [P[str(i)]
                       for i in range(17, 28)] + [jowl_bottom_point])
        Bl_p, Br_p = Bl / (Bl + Br), Br / (Bl + Br)

        utils.draw_text(image, f'{Bl_p * 100:.2f}%', jowl_bottom_point - np.array(
            [0.28, -0.1]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4)
        utils.draw_text(image, f'{Br_p * 100:.2f}%', jowl_bottom_point - np.array(
            [-0.17, -0.1]), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4)

        # Mouth areas
        mouth_top_point = line_intersection(
            middle_point, perp_point, P['78'], P['80'])
        mouth_bottom_point = line_intersection(
            middle_point, perp_point, P['86'], P['84'])
        Ml = poly_area([mouth_top_point] + [P[str(i)]
                       for i in range(78, 87)] + [mouth_bottom_point])
        Mr = poly_area([mouth_top_point] + [P[str(i)]
                       for i in range(80, 85)] + [mouth_bottom_point])
        Ml_p, Mr_p = Ml / (Ml + Mr), Mr / (Ml + Mr)

        utils.draw_text(image, f'{Ml_p * 100:.2f}%', mouth_top_point - np.array(
            [0.1, -0.04]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        utils.draw_text(image, f'{Mr_p * 100:.2f}%', mouth_top_point - np.array(
            [-0.01, -0.04]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        middle_point_0_32: np.ndarray = (P['32'] - P['0']) / 2
        middle_point_0_32 += P['0']

        # Eyes areas
        El = poly_area([P[str(i)] for i in range(60, 68)])
        Er = poly_area([P[str(i)] for i in range(68, 76)])
        El_p, Er_p = El / (El + Er), Er / (El + Er)

        utils.draw_text(image, f'{El_p * 100:.2f}%', P['64'] - np.array(
            [0.1, 0.04]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        utils.draw_text(image, f'{Er_p * 100:.2f}%', P['68'] - np.array(
            [-0.01, 0.04]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return image

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        angle = angle * 180 / np.pi - 180
        center = (image.shape[:2][0] / 2, image.shape[:2][1] / 2)
        img_rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1)
        res_img = cv2.warpAffine(
            image, img_rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return res_img

    def _calc_features(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        P = self.P
        # Calculate points on the line that divides the face in half
        ref1 = P[str(self.reference_points[0])]
        ref2 = P[str(self.reference_points[1])]
        middle_point: np.ndarray = (ref2 - ref1) / 2
        perp_point = middle_point.copy()
        perp_point[0], perp_point[1] = -middle_point[1], middle_point[0]
        middle_point += ref1
        perp_point += middle_point

        face_width = distance(P['0'], P['32'])
        features = {}
        features['f0'] = dist_to_line(
            middle_point, perp_point, P['51']) / face_width
        features['f1'] = dist_to_line(
            middle_point, perp_point, P['57']) / face_width  # !
        features['f2'] = dist_to_line(
            middle_point, perp_point, P['79']) / face_width
        features['f3'] = dist_to_line(
            middle_point, perp_point, P['85']) / face_width  # !
        features['f4'] = dist_to_line(
            middle_point, perp_point, P['16']) / face_width

        Ale = poly_area([P['60'], P['61'], P['62'], P['63'],
                        P['64'], P['65'], P['66'], P['67']])
        Are = poly_area([P['68'], P['69'], P['70'], P['71'],
                        P['72'], P['73'], P['74'], P['75']])
        features['f5'] = max(Ale/Are, Are/Ale)
        # features['f8'] = Ale/Are

        jowl_top_point = line_intersection(
            middle_point, perp_point, P['0'], P['32'])
        jowl_bottom_point = line_intersection(
            middle_point, perp_point, P['5'], P['27'])
        jaw_middle_point = line_intersection(
            middle_point, perp_point, P['15'], P['17'])

        # Jowl areas
        Ul = poly_area([jowl_top_point] + [P[str(i)]
                       for i in range(0, 6)] + [jowl_bottom_point])
        Ur = poly_area([jowl_bottom_point] + [P[str(i)]
                       for i in range(27, 33)] + [jowl_top_point])
        features['f6'] = max(Ul/Ur, Ur/Ul)
        # features['f9'] = Ul/Ur

        # Jaw areas
        Bl = poly_area([jowl_bottom_point] + [P[str(i)]
                       for i in range(5, 16)] + [jaw_middle_point])
        Br = poly_area([jaw_middle_point] + [P[str(i)]
                       for i in range(17, 28)] + [jowl_bottom_point])
        # features['f10'] = Bl/Br
        features['f7'] = max(Bl/Br, Br/Bl)

        # Mouth areas
        mouth_top_point = line_intersection(
            middle_point, perp_point, P['78'], P['80'])
        mouth_bottom_point = line_intersection(
            middle_point, perp_point, P['86'], P['84'])
        Ml = poly_area([mouth_top_point] + [P[str(i)]
                       for i in range(78, 87)] + [mouth_bottom_point])
        Mr = poly_area([mouth_top_point] + [P[str(i)]
                       for i in range(80, 85)] + [mouth_bottom_point])
        features['f8'] = max(Ml/Mr, Mr/Ml)
        # features['f11'] = Ml/Mr

        T = distance(P['60'], P['16'])
        U = distance(P['72'], P['16'])
        features['f9'] = max(T/U, U/T)
        # features['f0'] = T/U
        

        EPl = distance(P['60'], P['76'])
        EPr = distance(P['72'], P['82'])
        features['f10'] = max(EPl/EPr, EPr/EPl)

        ECl = distance(P['64'], P['76'])
        ECr = distance(P['68'], P['82'])
        features['f11'] = max(ECl/ECr, ECr/ECl)

        Wl = distance(P['76'], P['16'])
        Wr = distance(P['82'], P['16'])
        features['f12'] = max(Wl/Wr, Wr/Wl)

        ENl = distance(P['57'], P['60'])
        ENr = distance(P['57'], P['72'])
        features['f13'] = max(ENl/ENr, ENr/ENl)
        # BEl = distance(P['35'], P['66'])
        # BEr = distance(P['44'], P['74'])
        # features['f14'] = max(BEl/BEr, BEr/BEl)

        left_side_angles = [abs(angle(P[str(p2)], P[str(p1)]))
                            for p1, p2 in zip(range(16, 0, -1), range(15, -1, -1))]
        right_side_angles = [abs(angle(P[str(p1)], P[str(p2)]))
                            for p1, p2 in zip(range(16, 32), range(17, 33))]
        jaw_ratios = [max(r/l, l/r) for l, r in zip(left_side_angles, right_side_angles)]
        features['f14'] = max(jaw_ratios)

        # Calculate points at the center of a face and points on the perimeter
        middle_points = [middle_point, P['57']]

        left_side_points = [6, 10, 14]
        left_side_points = [P[f'{i}'] for i in left_side_points]
        right_side_points = [26, 22, 18]
        right_side_points = [P[f'{i}'] for i in right_side_points]

        perimeter_features = []
        for L, R in zip(left_side_points, right_side_points):
            for point in middle_points:
                dl = distance(point, L)
                dr = distance(point, R)
                perimeter_features.append(max(dl/dr, dr/dl))

        for i, feat in enumerate(perimeter_features):
            features[f'f{i + 15}'] = feat

        res = np.array(list(features.values()))
        # assert len(res) == self.geometric_features_num, "Wrong number of features"
        return res



class KarolewskiFilteredFeaturesExtractor(KarolewskiFeaturesExtractor):
    """
    Calculate features from the articles of Chang et al., Parra-Dominguez et al. and proposed by dr Karolewski. 
    By default points' indices are taken from the WFLW dataset annotation. 
    """

    def __init__(self, points_mapper: Optional[Callable[[int], int]] = None, reference_points: tuple[int, int] = (64, 68)):
        super().__init__(points_mapper, reference_points)
        self.geometric_features_num = 6

    def _calc_features(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        features = super()._calc_features(image, points)
        res = features[[4, 7, 8, 14, 16, 18]]
        assert len(res) == self.geometric_features_num, "Wrong number of features"
        return res

class EmptyFeaturesExtractor(DominguezFeaturesExtractor):

    def __init__(self, points_mapper: Optional[Callable[[int], int]] = None, reference_points: tuple[int, int] = (64, 68)):
        super().__init__(points_mapper=points_mapper, reference_points=reference_points)
        self.geometric_features_num = 0

    def _calc_features(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.array([])

