from scipy import spatial

# Feature embedding vector of enrolled faces
enrolled_faces = []

# The minimum distance between two faces
# to be called unique
authentication_threshold = 0.30


def enroll_face(embeddings):
    """
    This function adds the feature embedding
    for given face to the list of enrolled faces.
    This entire process is equivalent to
    face enrolment.
    """
    # Get feature embedding vector
    for embedding in embeddings:
        # Add feature embedding to list of
        # enrolled faces
        enrolled_faces.append(embedding)


def delist_face(embeddings):
    """
    This function removes a face from the list
    of enrolled faces.
    """
    # Get feature embedding vector for input images
    global enrolled_faces
    if len(embeddings) > 0:
        for embedding in embeddings:
            # List of faces remaining after delisting
            remaining_faces = []
            # Iterate over the enrolled faces
            for idx, face_emb in enumerate(enrolled_faces):
                # Compute distance between feature embedding
                # for input images and the current face's
                # feature embedding
                dist = spatial.distance.cosine(embedding, face_emb)
                # If the above distance is more than or equal to
                # threshold, then add the face to remaining faces list
                # Distance between feature embeddings
                # is equivalent to the difference between
                # two faces
                if dist >= authentication_threshold:
                    remaining_faces.append(face_emb)
            # Update the list of enrolled faces
            enrolled_faces = remaining_faces


def authenticate_emb(embedding):
    """
    This function checks if a similar face
    embedding is present in the list of
    enrolled faces or not.
    """
    # Set authentication to False by default
    authentication = False

    if embedding is not None:
        # Iterate over all the enrolled faces
        for face_emb in enrolled_faces:
            # Compute the distance between the enrolled face's
            # embedding vector and the input image's
            # embedding vector
            dist = spatial.distance.cosine(embedding, face_emb)
            # If above distance is less the threshold
            if dist < authentication_threshold:
                # Set the authenatication to True
                # meaning that the input face has been matched
                # to the current enrolled face
                authentication = True
        if authentication:
            # If the face was authenticated
            return True
        else:
            # If the face was not authenticated
            return False
    # Default
    return None
