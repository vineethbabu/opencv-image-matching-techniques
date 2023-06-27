import cv2
import os

def feature_matching(query_folder, images_folder):
    """
    Perform feature matching to find similarity between images in a query folder and a target folder.

    Parameters:
    - query_folder (str): Path to the folder containing query images.
    - images_folder (str): Path to the folder containing target images.

    Returns:
    - matches (list): List of dictionaries containing the best match for each query image.
                      Each dictionary contains the query image filename, match image filename,
                      and similarity score.
    """

    # Create feature detector and descriptor objects
    sift = cv2.SIFT_create()

    # List to store the similarity scores
    matches_final = []

    # Iterate over the images in the query folder
    for query_filename in os.listdir(query_folder):
        if query_filename.endswith('.jpg') or query_filename.endswith('.png'):
            # Load the query image from the query folder and convert to grayscale
            query_path = os.path.join(query_folder, query_filename)
            query_gray = cv2.imread(query_path, 0)

            # Detect and compute keypoints and descriptors for the query image
            query_keypoints, query_descriptors = sift.detectAndCompute(query_gray, None)

            # List to store the similarity scores for the current query image
            scores = []

            # Iterate over the images in the target folder
            for image_filename in os.listdir(images_folder):
                if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
                    # Load the target image from the target folder and convert to grayscale
                    image_path = os.path.join(images_folder, image_filename)
                    image_gray = cv2.imread(image_path, 0)

                    # Detect and compute keypoints and descriptors for the current target image
                    image_keypoints, image_descriptors = sift.detectAndCompute(image_gray, None)

                    # Create a matcher object
                    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

                    # Match the descriptors using the matcher
                    matches = matcher.knnMatch(query_descriptors, image_descriptors, k=2)

                    # Apply ratio test to filter good matches
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.9 * n.distance:
                            good_matches.append(m)

                    # Calculate the similarity score based on the number of good matches
                    similarity = len(good_matches) / max(len(query_keypoints), len(image_keypoints))

                    # Append the similarity score to the list
                    scores.append((image_filename, similarity))

            # Sort the scores based on the similarity in descending order
            scores.sort(key=lambda x: x[1], reverse=True)

            # Check if any match is found
            if scores:
                # Add the best match to the matches list
                best_match = {"query_image": query_filename, "match_image": scores[0][0], "similarity": scores[0][1]}
                matches_final.append(best_match)

            # Based on threshold value modify match image
            for ele in matches_final:
                if ele['similarity'] > 0.2 :
                    pass
                else:
                    ele['match_image'] = 'Not Found'

    return matches_final
