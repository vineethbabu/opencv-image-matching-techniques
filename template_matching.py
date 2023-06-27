import cv2
import os

def template_matching(query_folder, images_folder):

    """
    Perform template matching to find similarity between images in a query folder and a target folder.

    Parameters:
    - query_folder (str): Path to the folder containing query images.
    - images_folder (str): Path to the folder containing target images.

    Returns:
    - matches (list): List of dictionaries containing the best match for each query image.
                      Each dictionary contains the query image filename, match image filename,
                      and similarity score.
    """

    # List to store the similarity scores
    matches = []

    # Iterate over the images in the query folder
    for query_filename in os.listdir(query_folder):
        if query_filename.endswith('.jpg') or query_filename.endswith('.png'):
            # Load the query image from the query folder and convert to grayscale
            query_path = os.path.join(query_folder, query_filename)
            query_gray = cv2.imread(query_path, 0)

            # List to store the similarity scores for the current query image
            scores = []

            # Iterate over the images in the target folder
            for image_filename in os.listdir(images_folder):
                if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
                    # Load the target image from the target folder and convert to grayscale
                    image_path = os.path.join(images_folder, image_filename)
                    image_gray = cv2.imread(image_path, 0)

                    # Resize the template image to match the size of the target image
                    resized_template = cv2.resize(query_gray, (image_gray.shape[1], image_gray.shape[0]))

                    # Perform template matching
                    result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)

                    # Get the maximum correlation value from the result
                    similarity = cv2.minMaxLoc(result)[1]

                    # Append the similarity score to the list
                    scores.append((image_filename, similarity))

            # Sort the scores based on the similarity in descending order
            scores.sort(key=lambda x: x[1], reverse=True)

            # Check if any match is found
            if scores:
                # Add the best match to the matches list
                best_match = {"query_image": query_filename, "match_image": scores[0][0], "similarity": scores[0][1]}
                matches.append(best_match)
                
            # Based on threshold value modify match image
            for ele in matches:
                if ele['similarity'] > 0.5 :
                    pass
                else:
                    ele['match_image'] = 'Not Found'

    return matches
