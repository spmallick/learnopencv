
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from core.config import cfg
from core.dependencies import Dependencies
from core.Object_dependencies import Object_Detection
import datetime

cfg.all_model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def ob_detection():
    
    cap=cv2.VideoCapture(cfg.video_name)

    vehicles_entering = {}
    new_box_id = ' '

    while True:
        
        global veh_id,veh_value
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        if ret == False:
            break
        
        start_time = time.time()
        
        crp = Object_Detection(cfg.all_model,allowed_classes=cfg.allowed_classes,frame = frame)
        dependency = Dependencies(frame,cfg.allowed_classes)

        detections = dependency.deep_sort()
        
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        
        cfg.tracker.predict()
        cfg.tracker.update(detections)

        for track in cfg.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
        
            bbox = track.to_tlbr()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            vehicles_entering[(track.track_id)] = datetime.datetime.now()
            dependency.draw_functions(bbox,track.track_id,color)

            for box_id,_ in vehicles_entering.items():    
                pass

            crop_path = os.path.join(os.getcwd(), 'detections', 'crop')
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            
            final_path = os.path.join(crop_path, 'frame_'+ str(datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")))
            counted_classes = crp.count_objects(by_class = True)
        
            for obj_cateogry, obj_cout in counted_classes.items():
               
                if box_id != new_box_id:
                    new_box_id = box_id
                    try:
                        os.mkdir(final_path)
                    except FileExistsError:
                        pass          
                    crp.crop_objects(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),final_path)
 
                else:
                    pass    
            else:
                pass
            
            fps = 1.0 / (time.time() - start_time)
            # cv2.putText(frame, str("FPS: %.2f" % fps),(30,30),0, 0.75, (0,0,255),2)

            result = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)        
            cv2.imshow("result",result)

        key = cv2.waitKey(1)
        if key == ord('q'):

            break

    cap.release()
    cv2.destroyAllWindows()
    
ob_detection()