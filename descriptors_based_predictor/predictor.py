# Prediction based on quality of descriptor matches
# Accuracy: 74% paper, 75% rock, 94% scissors

def get_error_paper(img):
    final_error_paper = 0
    for t in range(10):
        paper_sample = cv2.imread("../data_manager/storage/raw/train/rps/paper/" + random.choice(os.listdir("../data_manager/storage/raw/train/rps/paper")))

        keypoint_detector = cv2.AKAZE_create()

        kp1, desc1 = keypoint_detector.detectAndCompute(paper_sample, None)
        kp2, desc2 = keypoint_detector.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key = lambda x:x.distance)
        error = 0
        best_matches = 10
        if len(matches) < best_matches:
            best_matches = len(matches)
        for i in range(best_matches):
            error += matches[i].distance
        final_error_paper += error / best_matches
    return final_error_paper

def get_error_rock(img):
    final_error_rock = 0

    for t in range(10):
        rock_sample = cv2.imread("../data_manager/storage/raw/train/rps/rock/" + random.choice(os.listdir("../data_manager/storage/raw/train/rps/rock")))
        keypoint_detector = cv2.AKAZE_create()
        kp1, desc1 = keypoint_detector.detectAndCompute(rock_sample, None)
        kp2, desc2 = keypoint_detector.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key = lambda x:x.distance)
        error = 0
        best_matches = 10
        if len(matches) < best_matches:
            best_matches = len(matches)
        for i in range(best_matches):
            error += matches[i].distance
        final_error_rock += error / best_matches
    return final_error_rock

def get_error_scissors(img):
    final_error_scissors = 0

    for t in range(10):
        scissors_sample = cv2.imread("../data_manager/storage/raw/train/rps/scissors/" + random.choice(os.listdir("../data_manager/storage/raw/train/rps/scissors")))
        keypoint_detector = cv2.AKAZE_create()

        kp1, desc1 = keypoint_detector.detectAndCompute(scissors_sample, None)
        kp2, desc2 = keypoint_detector.detectAndCompute(img, None)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key = lambda x:x.distance)

        error = 0
        best_matches = 10
        if len(matches) < best_matches:
            best_matches = len(matches)
        for i in range(best_matches):
            error += matches[i].distance
        final_error_scissors += error / best_matches
    return final_error_scissors

# input - image
# output - 1 for paper, 2 for rock, 3 for scissors
def get_all_errors(img):
    error_paper = get_error_paper(img)
    error_rock = get_error_rock(img)
    error_scissors = get_error_scissors(img)
    if error_paper <= error_rock and error_paper <= error_scissors:
        return 1
    elif error_rock <= error_scissors:
        return 2
    return 3