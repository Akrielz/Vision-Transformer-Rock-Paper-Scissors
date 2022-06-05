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


######################## SVM with EFD - 92.37% test accuracy ###########################

# Get EFD descriptors
def get_x_descriptor(img):
    r,g,b = cv2.split(img)
    r = 255 - r
    contours, _ = cv.findContours(r, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    fd = []
    for cnt in contours:
        coeffs = elliptic_fourier_descriptors(np.squeeze(cnt), order=10, normalize=True)
        fd.append(coeffs.flatten()[3:])
       
    return np.mean(np.array(fd), axis=0)

def prepare_data():
    X = []
    Y = []
    with os.scandir('./data_manager/storage/raw/train/rps/scissors/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            batch = img_generator.flow(np.reshape(image, (1, 300, 300, 3)), batch_size=1)
            
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(3)
                
                fdesc = get_x_descriptor(next(batch)[0].astype('uint8'))
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(3)
            except np.AxisError:
                continue # some corrupted images
    
    with os.scandir('./data_manager/storage/raw/train/rps/rock/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            batch = img_generator.flow(np.reshape(image, (1, 300, 300, 3)), batch_size=1)
            
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(2)
                
                fdesc = get_x_descriptor(next(batch)[0].astype('uint8'))
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(2)
            except np.AxisError:
                continue # some corrupted images
                
    with os.scandir('./data_manager/storage/raw/train/rps/paper/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            batch = img_generator.flow(np.reshape(image, (1, 300, 300, 3)), batch_size=1)
            
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(1)
                
                fdesc = get_x_descriptor(next(batch)[0].astype('uint8'))
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(1)
            except np.AxisError:
                continue # some corrupted images
    return np.array(X), np.array(Y)

def prepare_data_test():
    X = []
    Y = []
    with os.scandir('./data_manager/storage/raw/test/rps/scissors/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(3)
            except np.AxisError:
                continue # some corrupted images
    
    with os.scandir('./data_manager/storage/raw/test/rps/rock/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(2)
            except np.AxisError:
                continue # some corrupted images
                
    with os.scandir('./data_manager/storage/raw/test/rps/paper/') as it:
        for entry in it:
            image = cv2.imread(entry.path)
            try:
                fdesc = get_x_descriptor(image)
                X.append(copy.deepcopy(fdesc)) # pooling strategy here (task 3)
                Y.append(1)
            except np.AxisError:
                continue # some corrupted images
    return np.array(X), np.array(Y)

def get_accuracy():
    scaler = StandardScaler()
    x_train2 = scaler.fit_transform(x_train)
    x_test2 = scaler.transform(x_test)

    clf =  SVC(C=800000, kernel='rbf', gamma='auto')
    clf.fit(x_train2, y_train)
    predictions = clf.predict(x_test2)

    print("Classification accuracy: {}".format(accuracy_score(y_test, predictions)))
