import os


def is_image_large_enough(file_path):
    with Image.open(file_path) as img:
        return img.width >= MIN_WIDTH and img.height >= MIN_HEIGHT


def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((MIN_WIDTH, MIN_HEIGHT))
    img = img.convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    return img


def add_processed_images(df):
    df = df[df['file'].apply(is_image_large_enough)]
    df['image'] = df['file'].apply(preprocess_image)
    return df


# face detector
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'dlib/shape_predictor_68_face_landmarks.dat'
)


def contains_face(file_path):
    img = cv2.imread(file_path)

    if img is None:
        return False

    faces = face_detector(img, 1)
    return len(faces) > 0


def align_face(file_path):
    img = cv2.imread(file_path)

    if img is None:
        return None

    faces = face_detector(img, 1)

    if len(faces) == 0:
        return None

    shape = predictor(img, faces[0])
    aligned = dlib.get_face_chip(img, shape)

    return aligned


def normalize_face(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    return normalized


def augment_image(face):
    angle = np.random.uniform(-20, 20)

    M = cv2.getRotationMatrix2D(
        (face.shape[1] / 2, face.shape[0] / 2),
        angle,
        1
    )

    rotated = cv2.warpAffine(
        face,
        M,
        (face.shape[1], face.shape[0])
    )

    return rotated
