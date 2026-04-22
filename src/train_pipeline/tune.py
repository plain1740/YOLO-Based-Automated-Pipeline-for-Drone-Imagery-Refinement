from ultralytics import YOLO
def main():
    model = YOLO("yolo26s.pt")
    model.tune(
         data="TRAIN_DATA3/data.yaml",
         iterations=50,
         epochs=200,
         batch=0.9,
         plots=False,
         workers=6
     )
if __name__=="__main__":
    main()

