# Standard library imports

# Third party imports
import cv2

# Local application imports
from functions import ScannerComponent, qr_scanner, qr_decoder


def video_handler(args):
    """
    handler for video command
    """
    verbose = args.verbose
    thresh = args.thresh
    continuous_detection = args.continuous

    if args.filename:
        video_source = args.filename
        scanner_view = None
    else:
        video_source = args.device
        scanner_view = ScannerComponent(3, (0.4, 0.6))

    cap = cv2.VideoCapture(video_source)
    qr_old_data = ''

    while True:

        ret, input_image = cap.read()
        if not ret:
            break

        if scanner_view:
            scanner_view.set_input_frame(input_image)
            objective_image = scanner_view.get_central_image()
            drawn_scanner = scanner_view.draw()
            cv2.imshow("QRScanner", drawn_scanner)
        else:
            objective_image = input_image

        qr_detected = qr_scanner(objective_image, thresh, verbose)
        if qr_detected:
            cv2.imshow("QR Code Detected", qr_detected[0])
            qr_data = qr_decoder(qr_detected[0])

            if verbose:
                cv2.imshow("Input Image", objective_image)
                cv2.imshow("Placeholders", qr_detected[1])

            if qr_data and (qr_old_data != qr_data):
                print(f"data: {qr_data}")
                qr_old_data = qr_data

            if not continuous_detection:
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if continuous_detection or cv2.waitKey(0):
        cap.release()
        cv2.destroyAllWindows()


def image_handler(args):
    """
    handler for image command
    """

    path_to_image = args.path
    verbose = args.verbose
    thresh = args.thresh

    img = cv2.imread(path_to_image)
    if np.any(img):
        qr_detected = qr_scanner(img, thresh, verbose)
        if qr_detected:

            qr_data = qr_decoder(qr_detected[0])
            print(f"data: {qr_data}")

            if verbose:
                cv2.imshow("Input Image", img)
                cv2.imshow("Placeholders", qr_detected[1])
                cv2.imshow("QR Code Detected", qr_detected[0])

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
