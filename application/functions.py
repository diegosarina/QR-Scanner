# Standard library imports

# Third party imports
import numpy as np
import cv2
import pyzbar.pyzbar as pyzbar

# Local application imports

# Constants variables
QR_NORTH = 0
QR_EAST = 1
QR_SOUTH = 2
QR_WEST = 3

OTSU_BINARIZATION = 0
THRES_BINARIZATION = 1

DEBUG_MODE_OFF = 0
DEBUG_MODE_ON = 1


class PointIntersectionError(Exception):
    pass

class OffLineError(Exception):
    pass


def distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def distance_line_to_point(line_p1, line_p2, p):
    m = (line_p2[1] - line_p1[1]) / (line_p2[0] - line_p1[0])
    a = -m
    b = 1.0
    c = (m * line_p1[0]) - line_p1[1]
    dist = (a * p[0] + b * p[1] + c) / np.sqrt((a * a)+(b * b))

    return dist


def line_slope(point_1, point_2):
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    if dy:
        return (dy / dx), 1

    return 0, 0


def update_corner(c1, ref, baseline, corner):
    temp_dist = distance(c1, ref)

    if temp_dist > baseline:
        baseline = temp_dist
        corner = c1
        return baseline, corner

    return baseline, corner


def get_vertices(contour, image, draw=False):
    """
        A-----W-----B
        |           |
        Z           X
        |           |
        D-----Y-----C
    """
    box = cv2.boundingRect(contour)
    A = box[0:2]
    B = [A[0] + box[2], A[1]]
    C = [B[0], A[1] + box[3]]
    D = [A[0], C[1]]

    dmax = np.array([0, 0, 0, 0])
    M0, M1, M2, M3 = [0, 0], [0, 0], [0, 0], [0, 0]
    half_x = (A[0] + B[0]) / 2
    half_y = (A[1] + D[1]) / 2
    if draw:
        image[int(half_y), int(half_x), :] = (0, 0, 255)
        vertices = [A, B, C, D]
        for p in vertices:
            image[p[1], p[0], :] = (255, 0, 0)

    for point in contour:
        if draw:
            image[point[0][1], point[0][0], :] = (0, 255, 0)
            vertices = [M0, M1, M2, M3]
            for p in vertices:
                image[p[1], p[0], :] = (0, 0, 255)
            cv2.imshow('get_vertices', image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                draw = False

        if (point[0][0] < half_x) and (point[0][1] <= half_y):
            dmax[2], M0 = update_corner(point[0], C, dmax[2], M0)

        elif (point[0][0] >= half_x) and (point[0][1] < half_y):
            dmax[3], M1 = update_corner(point[0], D, dmax[3], M1)

        elif (point[0][0] > half_x) and (point[0][1] >= half_y):
            dmax[0], M2 = update_corner(point[0], A, dmax[0], M2)

        elif (point[0][0] <= half_x) and (point[0][1] > half_y):
            dmax[1], M3 = update_corner(point[0], B, dmax[1], M3)

    if draw:
        cv2.destroyWindow('get_vertices')
    return [M0, M1, M2, M3]


def update_corner_orientation(orientation, vertices):

    return {
        QR_NORTH: [vertices[0], vertices[1], vertices[2], vertices[3]],
        QR_EAST: [vertices[1], vertices[2], vertices[3], vertices[0]],
        QR_SOUTH: [vertices[2], vertices[3], vertices[0], vertices[1]],
        QR_WEST: [vertices[3], vertices[0], vertices[1], vertices[2]]
    }[orientation]


def intersection_between_two_lines(line1_p1, line1_p2, line2_p1, line2_p2):
    delta_x = (line1_p2[0] - line1_p1[0], line2_p2[0] - line2_p1[0])
    delta_y = (line1_p2[1] - line1_p1[1], line2_p2[1] - line2_p1[1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    determinant = det(delta_x, delta_y)
    if not determinant:
        raise PointIntersectionError('lines do not intersect')

    d = (det(line1_p2, line1_p1), det(line2_p2, line2_p1))
    x = float(det(d, delta_x)) / determinant
    y = float(det(d, delta_y)) / determinant

    return np.int32([x, y])


def image_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 220, (9, 9), L2gradient=True)
    return edges


def mass_center(contours):
    mass_centers = np.empty([len(contours), 2])
    for i, contour in enumerate(contours):
        moments = cv2.moments(contour)
        if moments["m00"]:
            mass_centers[i, :] = (
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"]
            )
        else:
            mass_centers[i, :] = (0, 0)

    return mass_centers


def search_candidate_markers(contours, hierarchy):
    marks_candidates = []

    for i, contour in enumerate(contours):
        approx = cv2.approxPolyDP(
            contour,
            cv2.arcLength(contour, True) * 0.02,
            True
        )
        if len(approx) == 4:
            k = i
            cc = 0

            while (hierarchy[0, k, 2] != -1):
                k = hierarchy[0, k, 2]
                cc += 1

            if (hierarchy[0, k, 2] != -1):
                cc += 1

            if cc >= 5:
                marks_candidates.append(i)

    return marks_candidates


def find_correct_mark(marks_candidates):
    A = marks_candidates[0]
    B = marks_candidates[1]
    C = marks_candidates[2]
    return A, B, C


def find_offline_point(mass_centers, A, B, C):
    AB = distance(mass_centers[A, :], mass_centers[B, :])
    BC = distance(mass_centers[B, :], mass_centers[C, :])
    CA = distance(mass_centers[C, :], mass_centers[A, :])

    if CA < AB > BC:
        offline_point = C
        return offline_point, A, B
    if BC < CA > AB:
        offline_point = B
        return offline_point, A, C
    if CA < BC > AB:
        offline_point = A
        return offline_point, B, C

    raise OffLineError


def get_orientation(align, slope, dist, mass_centers, mediana_1, mediana_2):
    if not align:
        botton = mediana_1
        rigth = mediana_2
    elif slope < 0 and dist < 0:
        orientation = QR_NORTH
        if mass_centers[mediana_1][1] > mass_centers[mediana_2][1]:
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1
    elif dist < 0 < slope:
        orientation = QR_EAST
        if mass_centers[mediana_1][0] > mass_centers[mediana_2][0]:
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif slope < 0 < dist:
        orientation = QR_SOUTH
        if mass_centers[mediana_1][1] > mass_centers[mediana_2][1]:
            botton = mediana_2
            rigth = mediana_1
        else:
            botton = mediana_1
            rigth = mediana_2
    elif slope > 0 and dist > 0:
        orientation = QR_WEST
        if mass_centers[mediana_1][0] > mass_centers[mediana_2][0]:
            botton = mediana_1
            rigth = mediana_2
        else:
            botton = mediana_2
            rigth = mediana_1

    return orientation, botton, rigth


def perspective_correction(vertices, image):

    dst = np.array(((40, 40), (440, 40), (440, 440), (40, 440)), dtype=np.float32)
    warp_matrix = cv2.getPerspectiveTransform(vertices, dst)
    warped = cv2.warpPerspective(
        image,
        warp_matrix,
        (500, 500),
        borderMode=cv2.BORDER_REPLICATE
    )
    image = cv2.copyMakeBorder(
        warped,
        10, 10, 10, 10,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )
    return image


def binarize_image(image, thresh=-1):

    if thresh == -1:
        _, otsu = cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            0, 255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU
        )
        return otsu

    _, binary = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        thresh, 255,
        cv2.THRESH_BINARY
    )
    return binary


def plot_markers(image, contours, top, rigth, bottom, N, tickness=1):

    cv2.drawContours(image, contours, top, (255, 200, 0), tickness)
    cv2.drawContours(image, contours, rigth, (0, 0, 255), tickness)
    cv2.drawContours(image, contours, bottom, (255, 0, 100), tickness)

    return image


def plot_vertices(image, L, M, O, N):
    cv2.circle(image, tuple(L[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(L[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(L[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(image, tuple(L[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(image, tuple(M[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(M[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(M[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(image, tuple(M[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(image, tuple(O[0]), 2, (255, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(O[1]), 2, (0, 255, 0), -1, 8, 0)
    cv2.circle(image, tuple(O[2]), 2, (0, 0, 255), -1, 8, 0)
    cv2.circle(image, tuple(O[3]), 2, (128, 128, 128), -1, 8, 0)

    cv2.circle(image, tuple(N), 5, (70, 252, 252), -1, 8, 0)


def plot_lines(image, M, O, N):
    cv2.line(image, tuple(M[1]), tuple(N), (0, 0, 255), 1, 8, 0)
    cv2.line(image, tuple(O[3]), tuple(N), (0, 0, 255), 1, 8, 0)


def qr_scanner(image, thresh=-1, verbose=False):
    ret = ()
    edges = image_edges(image)

    #APPROX SIMPLE -> SE QUEDA CON LOS PUNTOS MAS EXTERNOS DEL CONTORNO
    #APPROX NONE -> DEVUELVE TODOS LOS PUNTOS
    contours, herencia = cv2.findContours(
        np.copy(edges),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_TC89_KCOS
    )
    mass_centers = mass_center(contours)
    marks_candidates = search_candidate_markers(contours, herencia)

    if len(marks_candidates) >= 3: #marcadores descubiertos

        A, B, C = find_correct_mark(marks_candidates)
        area_A = cv2.contourArea(contours[A])
        area_B = cv2.contourArea(contours[B])
        area_C = cv2.contourArea(contours[C])

        average_area = (area_A + area_B + area_C) / 3

        if not ((average_area * 0.8 < area_A < average_area * 1.2)
                and (average_area * 0.8 < area_B < average_area * 1.2)
                and (average_area * 0.8 < area_C < average_area * 1.2)):
            return

        try:
            top, mediana_1, mediana_2 = find_offline_point(mass_centers, A, B, C)
        except OffLineError:
            return

        dist = distance_line_to_point(
            mass_centers[mediana_1],
            mass_centers[mediana_2],
            mass_centers[top]
        )
        slope, align = line_slope(
            mass_centers[mediana_1],
            mass_centers[mediana_2]
        )

        orientation, bottom, rigth = get_orientation(
            align,
            slope,
            dist,
            mass_centers,
            mediana_1,
            mediana_2
        )

        if (cv2.contourArea(contours[top]) > 10
                and cv2.contourArea(contours[rigth]) > 10
                and cv2.contourArea(contours[bottom]) > 10):
            image_point = np.copy(image)
            L = update_corner_orientation(
                orientation,
                get_vertices(contours[top], image_point, verbose)
            )
            M = update_corner_orientation(
                orientation,
                get_vertices(contours[rigth], image_point, verbose)
            )
            O = update_corner_orientation(
                orientation,
                get_vertices(contours[bottom], image_point, verbose)
            )
            try:
                N = intersection_between_two_lines(M[1], M[2], O[3], O[2])
            except PointIntersectionError:
                return None
            try:
                pixel = image[N[1], N[0]]
            except:
                return

            external_vertices = np.array(([L[0], M[1], N, O[3]]), dtype=np.float32)
            corrected_perspective = perspective_correction(
                external_vertices,
                image
            )

            ret = (
                binarize_image(
                    corrected_perspective,
                    thresh
                ),
                plot_markers(
                    np.copy(image),
                    contours,
                    top, rigth,
                    bottom, N,
                    tickness=3
                )
            )

            if verbose:
                traces = np.zeros((np.shape(image)), dtype=np.uint8)
                cv2.drawContours(traces, contours, top, (255, 0, 100), 1)
                cv2.drawContours(traces, contours, rigth, (255, 0, 100), 1)
                cv2.drawContours(traces, contours, bottom, (255, 0, 100), 1)

                plot_vertices(traces, L, M, O, N)
                plot_lines(traces, M, O, N)

                orientations = {
                    QR_NORTH : "NORTH",
                    QR_SOUTH : "SOUTH",
                    QR_EAST : "EAST",
                    QR_WEST : "WEST",
                }
                cv2.putText(
                    traces,
                    orientations[orientation],
                    (20, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0), 1, 8
                )
                cv2.imshow("traces", traces)

    return ret


def qr_decoder(qr_codificado):
    decoded_data = None
    qrs = pyzbar.decode(qr_codificado)

    for qr in qrs:
        decoded_data = qr.data.decode("utf-8")

    return decoded_data


class ScannerComponent():
    def __init__(self, line_speed, percentage):
        self._percentage = percentage
        self._line_speed = line_speed
        self._iter = 0

    def set_input_frame(self, frame):
        middle_x = int(frame.shape[1] / 2)
        middle_y = int(frame.shape[0] / 2)

        delta_x = int((frame.shape[1] * self._percentage[0]) / 2)
        delta_y = int((frame.shape[0] * self._percentage[1]) / 2)

        self.min_x = middle_x - delta_x
        self.max_x = middle_x + delta_x
        self.min_y = middle_y - delta_y
        self.max_y = middle_y + delta_y

        self.frame = frame
        self._central_image = np.copy(
            frame[
                self.min_y:self.max_y,
                self.min_x:self.max_x
            ]
        )

    def get_central_image(self):
        return self._central_image

    def draw(self):
        background = cv2.GaussianBlur(self.frame, (25,25), 5)
        background[
            self.min_y:self.max_y,
            self.min_x:self.max_x
        ] = self._central_image

        rectangle_color = (169, 169, 172)
        line_color = (0, 0, 255)

        cv2.rectangle(
            background,
            (self.min_x, self.min_y),
            (self.max_x, self.max_y),
            rectangle_color,
            1
        )
        cv2.line(
            background,
            (self.min_x, self._iter),
            (self.max_x, self._iter),
            line_color,
            1
        )
        self._increase_iterator()
        return background

    def _increase_iterator(self):
        if(self._iter >= self.max_y or self._iter <= self.min_y):
            self._iter = self.min_y
        self._iter += self._line_speed
