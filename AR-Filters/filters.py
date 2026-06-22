
import cv2
import numpy as np
import mediapipe as mp


# Configuration Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh =  mp_face_mesh.FaceMesh(
min_detection_confidence=0.5,
min_tracking_confidence=0.5)


# Import all filters
mustache = cv2.imread('filters/mustache.png', cv2.IMREAD_UNCHANGED)
glasses = cv2.imread('filters/glasses.png', cv2.IMREAD_UNCHANGED)
squidgame = cv2.imread('filters/Squid-Game-Guard-Mask.png', cv2.IMREAD_UNCHANGED) 
frontman = cv2.imread('filters/Squid-Game-Front-Man-Mask.png', cv2.IMREAD_UNCHANGED) 
mask = cv2.imread('filters/mask.png', cv2.IMREAD_UNCHANGED)
anime = cv2.imread('filters/anime.png', cv2.IMREAD_UNCHANGED)
anonymous = cv2.imread('filters/anonymous.png', cv2.IMREAD_UNCHANGED)
santahat =  cv2.imread('filters/santa-hat.png', cv2.IMREAD_UNCHANGED)
beard =  cv2.imread('filters/beard.png', cv2.IMREAD_UNCHANGED)


# Find intersection point for two lines
def find_intersection(line1, line2):
    '''
        line format: y = line[0]x + line[1]
    '''
    x = ((line2[1] - line1[1]) / (line1[0] - line2[0])) 
    y = ((line1[0] * line2[1]) - (line2[0] * line1[1])) / (line1[0] - line2[0])
    return [x, y]

# Find line with slope from point1 & point2 and passes through point3
def find_line(point1, point2, point3):
    m = (point2[1] - point1[1]) / (point2[0] - point1[0]) 
    c = (point3[1]) - (m * point3[0])
    return [m, c]

def find_dist(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

'''
Function to apply add a full mask filter
'''
def apply_filter(source, imageFace, dstMat):
    (img_h, img_w) = imageFace.shape[:2]

    (srcH, srcW) = source.shape[:2]          
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # compute the homography matrix and then warp the source image to the
    # destination based on the homography
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (img_w, img_h))


    # Split out the transparency mask from the colour info
    overlay_img = warped[:,:,:3] # Grab the BRG planes
    overlay_mask = warped[:,:,3:]  # And the alpha plane

    # Calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to inting point in range 0.0 - 1.0
    face_part = (imageFace * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    output = np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    return output


def points(results, image,  Filter):
    vpoint1 = [results[0].landmark[10].x * image.shape[1], results[0].landmark[10].y * image.shape[0]]
    vpoint2 = [results[0].landmark[152].x * image.shape[1], results[0].landmark[152].y * image.shape[0]]
    hpoint1 = [results[0].landmark[234].x * image.shape[1], results[0].landmark[234].y * image.shape[0]]
    hpoint2 = [results[0].landmark[454].x * image.shape[1], results[0].landmark[454].y * image.shape[0]]

    hscale = find_dist(hpoint1, hpoint2)
    vscale = find_dist(vpoint1, vpoint2)
    if  Filter == "full_face":
        # Transform matrix
        
        hline1 = find_line(hpoint1, hpoint2, vpoint1)
        vline1 = find_line(vpoint1, vpoint2, hpoint1)
        vline2 = find_line(vpoint1, vpoint2, hpoint2)
        hline2 = find_line(hpoint1, hpoint2, vpoint2)

        topLeft = [find_intersection(vline2, hline1)[0] + 0.05 * hscale, find_intersection(vline2, hline1)[1] - 0.12 * vscale]
        bottomRight = [find_intersection(vline1, hline2)[0] - 0.05 * hscale, find_intersection(vline1, hline2)[1] + 0.1 * vscale]
        topRight = [find_intersection(vline1, hline1)[0] - 0.05 * hscale, find_intersection(vline1, hline1)[1] - 0.12 * vscale]
        bottomLeft = [find_intersection(vline2, hline2)[0] + 0.05 * hscale, find_intersection(vline2, hline2)[1] + 0.1 * vscale]
        

                
    elif  Filter == "mask":
        
        point3 = [results[0].landmark[168].x * image.shape[1], results[0].landmark[168].y * image.shape[0]]

        hline1 = find_line(hpoint1, hpoint2, point3)
        vline1 = find_line(vpoint1, vpoint2, hpoint1)
        vline2 = find_line(vpoint1, vpoint2, hpoint2)
        hline2 = find_line(hpoint1, hpoint2, vpoint2)

        topLeft = [find_intersection(vline2, hline1)[0] + 0.05 * hscale, find_intersection(vline2, hline1)[1]]
        bottomRight = [find_intersection(vline1, hline2)[0] - 0.05 * hscale, find_intersection(vline1, hline2)[1] + 0.05 * vscale]
        topRight = [find_intersection(vline1, hline1)[0] - 0.05 * hscale, find_intersection(vline1, hline1)[1]]
        bottomLeft = [find_intersection(vline2, hline2)[0] + 0.05 * hscale, find_intersection(vline2, hline2)[1] + 0.05 * vscale]
        
        
    elif  Filter == "mustache":
        
        topLeft = [results[0].landmark[205].x * image.shape[1], results[0].landmark[205].y * image.shape[0]]
        topRight = [results[0].landmark[425].x * image.shape[1], results[0].landmark[425].y * image.shape[0]]
        bottomRight = [results[0].landmark[436].x * image.shape[1], results[0].landmark[436].y * image.shape[0]]
        bottomLeft = [results[0].landmark[216].x * image.shape[1], results[0].landmark[216].y * image.shape[0]]


    elif  Filter == "eyes":
        
        topLeft = [results[0].landmark[21].x * image.shape[1], results[0].landmark[21].y * image.shape[0]]
        topRight = [results[0].landmark[251].x * image.shape[1], results[0].landmark[251].y * image.shape[0]]
        bottomRight = [results[0].landmark[323].x * image.shape[1], results[0].landmark[323].y * image.shape[0]]
        bottomLeft = [results[0].landmark[93].x * image.shape[1], results[0].landmark[93].y * image.shape[0]]
    
    elif Filter == "hat":
        vpoint3 = [results[0].landmark[10].x * image.shape[1], results[0].landmark[10].y * image.shape[0] - 1.1*vscale]
        vpoint1 = [results[0].landmark[151].x * image.shape[1], results[0].landmark[151].y * image.shape[0]]
        hline1 = find_line(hpoint1, hpoint2, vpoint1)
        vline1 = find_line(vpoint1, vpoint2, hpoint1)
        vline2 = find_line(vpoint1, vpoint2, hpoint2)
        hline2 = find_line(hpoint1, hpoint2, vpoint3)

        bottomLeft = [find_intersection(vline2, hline1)[0] + 0.2 * hscale, find_intersection(vline2, hline1)[1]]
        bottomRight = [find_intersection(vline1, hline1)[0] - 0.2 * hscale, find_intersection(vline1, hline1)[1]]
        topLeft = [find_intersection(vline2, hline2)[0] + 0.2 * hscale, find_intersection(vline2, hline2)[1]]
        topRight = [find_intersection(vline1, hline2)[0] - 0.2 * hscale, find_intersection(vline1, hline2)[1]]
        
    elif Filter == "beard":
        vpoint1 = [results[0].landmark[5].x * image.shape[1], results[0].landmark[5].y * image.shape[0]]
        vpoint2 = [results[0].landmark[152].x * image.shape[1], results[0].landmark[152].y * image.shape[0]+0.2*vscale]    
        hline1 = find_line(hpoint1, hpoint2, vpoint1)
        vline1 = find_line(vpoint1, vpoint2, hpoint1)
        vline2 = find_line(vpoint1, vpoint2, hpoint2)
        hline2 = find_line(hpoint1, hpoint2, vpoint2)

        topLeft = [find_intersection(vline2, hline1)[0] + 0.1 * hscale, find_intersection(vline2, hline1)[1]+0.05*vscale]
        bottomRight = [find_intersection(vline1, hline2)[0] - 0.1 * hscale, find_intersection(vline1, hline2)[1]]
        topRight = [find_intersection(vline1, hline1)[0] - 0.1 * hscale, find_intersection(vline1, hline1)[1]+0.05*vscale]
        bottomLeft = [find_intersection(vline2, hline2)[0] + 0.1 * hscale, find_intersection(vline2, hline2)[1]]
         

    dstMat = [ topLeft, topRight, bottomRight, bottomLeft ]
    dstMat = np.array(dstMat)
    return dstMat


def transform(img, type):

    image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    # To improve performance
    image.flags.writeable = False
    results =  face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    resultImg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        if  type != "map points":
            
            if  type == "mustache":
                Filter = "mustache"
                filterImg =  mustache

            elif  type == "glasses":
                Filter = "eyes"
                filterImg =  glasses

            elif  type == "squidgame":
                Filter = "full_face"
                filterImg =  squidgame

            elif  type == "front man":
                Filter = "full_face"
                filterImg =  frontman

            elif  type == "mask":
                Filter = "mask"
                filterImg =  mask

            elif  type == "anime":
                Filter = "full_face"
                filterImg =  anime

            elif  type == "anonymous":
                Filter = "full_face"
                filterImg =  anonymous

            elif type == "santa hat":
                Filter = "hat"
                filterImg = santahat

            elif type == "beard":
                Filter = "beard"
                filterImg = beard
            
            elif type == "santa":
                Filter = "beard"
                filterImg = beard
                dstMat =   points(results.multi_face_landmarks, image,  Filter) 
                resultImg =  apply_filter(filterImg,resultImg,dstMat)
                Filter = "hat"
                filterImg = santahat



            dstMat =   points(results.multi_face_landmarks, image,  Filter) 
            resultImg =  apply_filter(filterImg,resultImg,dstMat)  
            
        else:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec =  mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                    image=resultImg,
                    landmark_list=face_landmarks,
                    connections= mp_face_mesh. FACEMESH_CONTOURS,
                    landmark_drawing_spec= drawing_spec,
                    connection_drawing_spec= drawing_spec)
    return resultImg

if __name__ == "__main__":
    vid = cv2.VideoCapture(0)
    filtername = 'map points'
    while(True):
        
        ret, frame = vid.read()
        op = transform(frame, filtername)

        # Display the resulting frame
        cv2.imshow('1-9: filters, 0: map-points & s: Christmas Vibes', op)     
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('1'):
            filtername = 'anonymous'
        elif k == ord('2'):
            filtername = 'anime'
        elif k == ord('3'):
            filtername = 'mask'
        elif k == ord('4'):
            filtername = 'front man'
        elif k == ord('5'):
            filtername = 'squidgame'
        elif k == ord('6'):
            filtername = 'glasses'
        elif k == ord('7'):
            filtername = 'mustache'
        elif k == ord('8'):
            filtername = 'santa hat'
        elif k == ord('9'):
            filtername = 'beard'
        elif k == ord('0'):
            filtername = 'map points'
        elif k == ord('s'):
            filtername = 'santa'
        
        