import numpy as np
import cv2
import os
import imutils

"""
Comenzamos Capturando los Rostros
que deseamos entrenar
"""
class Reconocimiento_Facial():
    def __init__(self, arg):
        super(Reconocimiento_Facial, self).__init__()
        
    def highlightFace(net, img, conf_threshold=0.7):
        imagenDnn=img.copy()
        imgHeight=imagenDnn.shape[0]
        imgWidth=imagenDnn.shape[1]
        blob=cv2.dnn.blobFromImage(imagenDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections=net.forward()
        faceBoxes=[]
        for i in range(detections.shape[2]):
            confidence=detections[0,0,i,2]
            if confidence>conf_threshold:
                x1=int(detections[0,0,i,3]*imgWidth)
                y1=int(detections[0,0,i,4]*imgHeight)
                x2=int(detections[0,0,i,5]*imgWidth)
                y2=int(detections[0,0,i,6]*imgHeight)
                faceBoxes.append([x1,y1,x2,y2])
                cv2.rectangle(imagenDnn, (x1,y1), (x2,y2), (0,255,0), int(round(imgHeight/150)), 1)
        return imagenDnn,faceBoxes

    def Capturando_Rostros(img):
        global count
        global rostro_cascade        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = img.copy()
        faces = rostro_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = image[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(ruta_persona + '/rostro_{}.jpg'.format(count),rostro)
            count = count + 1

        return image

    def Entrenamiento_Rostros():
        global Lista_persona
        print('Lista de personas: ', Lista_persona)
        labels = []
        facesData = []
        label = 0

        for Nombre_Carp in Lista_persona:
            ruta_nombre = ruta + '/' + Nombre_Carp
            print('Leyendo las imágenes')

            for fileName in os.listdir(ruta_nombre):
                print('Rostro: ', Nombre_Carp + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(ruta_nombre+'/'+fileName,0))
            label = label + 1

        reconocedor_caras = cv2.face.LBPHFaceRecognizer_create()
        print("Entrenando...")
        reconocedor_caras.train(facesData, np.array(labels))
        reconocedor_caras.write('modeloLBPHFace.xml')
        print("Modelo almacenado...")
        return False
    def Detectando_Rostros(img,faceNet):
        global rostro_cascade
        global reconocedor_caras
        global padding

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = gray.copy()

        faces = rostro_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            rostro = image[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            resultado = reconocedor_caras.predict(rostro)

            #cv2.putText(img,'{}'.format(resultado),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            img,faceBoxes=Reconocimiento_Facial.highlightFace(faceNet,img)
            if not faceBoxes:
                print("Cara no Detectada")


            for faceBox in faceBoxes:
                face=img[max(0,faceBox[1]-padding):
                           min(faceBox[3]+padding,img.shape[0]-1),max(0,faceBox[0]-padding)
                           :min(faceBox[2]+padding, img.shape[1]-1)]

                #prediccion de sexo
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), HOMBRE_VALORES, swapRB=False)
                generoNet.setInput(blob)
                genderPreds=generoNet.forward()
                gender=genero_lista[genderPreds[0].argmax()]
                #prediccion de edad
                edadNet.setInput(blob)
                agePreds=edadNet.forward()
                age=edad_lista[agePreds[0].argmax()]

                if resultado[1] < 70:
                    #cv2.putText(img,'{}'.format(Lista_persona[resultado[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    
                    cv2.putText(img, f'Nombre:  {Lista_persona[resultado[0]]}', (faceBox[0]-40, faceBox[1]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
                    cv2.putText(img, f'Sexo: , {gender}', (faceBox[0]-40, faceBox[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
                    cv2.putText(img, f'Edad: {age}', (faceBox[0]-40, faceBox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)
           
                    print(f"Nombre: {Lista_persona[resultado[0]]}, Sexo: {gender}, Edad: {age}.")
                #print('{}'.format(resultado[1]))
                else:
                    cv2.putText(img,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(img, (x,y),(x+w,y+h),(0,0,255),2)
            return img


faceProto="/modelo/deploy.prototxt"
faceModel="/modelo/res10_300x300_ssd_iter_140000_fp16.caffemodel"
proto_edad="/modelo/age_deploy.prototxt"
modelo_edad="/modelo/age_net.caffemodel"
proto_genero="/modelo/gender_deploy.prototxt"
modelo_genero="/modelo/gender_net.caffemodel"

HOMBRE_VALORES=(78.4263377603, 87.7689143744, 114.895847746)
edad_lista=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genero_lista=['Hombre','Mujer']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
edadNet=cv2.dnn.readNet(modelo_edad,proto_edad)
generoNet=cv2.dnn.readNet(modelo_genero,proto_genero)
padding = 20
ans=True
while ans:
    ruta = '/reconocimiento_facial/data'
    rostro_cascade = cv2.CascadeClassifier('/modelo/haarcascade_frontalface_default.xml')
    count = 0
    print ("""
    1.Capturar Rostros
    2.Entrenar Rostros Capturados
    3.Reconocimiento Facial
    4.Exit/Quit
    """)
    ans=input("Ingresa la opcion que desees: ") 
    if ans=="1":
        cap = cv2.VideoCapture(0, cv2.CAP_V4L) 
        Nombre = input("Ingresa Tu Nombre: ")
        ruta_persona = ruta + '/' + Nombre
        if not os.path.exists(ruta_persona):
            print('Carpeta creada: ',ruta_persona)
            os.makedirs(ruta_persona)
        
        while True:
            ret, img = cap.read()
            img = imutils.resize(img,width=800)
            if ret == False:
                break
            img_detection = Reconocimiento_Facial.Capturando_Rostros(img)

            cv2.imshow('Capturando Rostros',img_detection)
            
            if cv2.waitKey(1) == ord('a'):
                break
        cap.release()
        cv2.destroyAllWindows()        
    elif ans=="2":
      Lista_persona = os.listdir(ruta)
      Reconocimiento_Facial.Entrenamiento_Rostros()
    elif ans=="3":
        Lista_persona = os.listdir(ruta)
        reconocedor_caras = cv2.face.LBPHFaceRecognizer_create()
        reconocedor_caras.read('modeloLBPHFace.xml')
        cap = cv2.VideoCapture(0, cv2.CAP_V4L)         
        while True:
            ret, img = cap.read()
            img = imutils.resize(img,width=800)
            if ret == False:
                break
            img_detection = Reconocimiento_Facial.Detectando_Rostros(img,faceNet)


            cv2.imshow('Detectando Rostros',img_detection)
            
    
            if cv2.waitKey(1) == ord('a'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif ans=="4":
      print("\n Goodbye")
      ans=False 
    elif ans !="":
      print("\n Ingresa una Opcion valida") 
    




