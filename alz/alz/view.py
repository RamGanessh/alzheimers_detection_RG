from posixpath import abspath
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os


import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from keras.models import load_model

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request,'predict.html')

def res(request):
    
    fobj=request.FILES["img"]
    fs=FileSystemStorage()
    img=fs.save(fobj.name,fobj)
    path=fs.url(img)


    abpath=os.path.abspath(__file__)
    abpath=os.path.dirname(abpath)
    abpath=abpath.replace("\\","/")
    var=abpath+'/media/'+img    


    context={"imgname":img, "imgpath":path}

    model=load_model("alzheimers125.h5")

    img=image.load_img(var,target_size=(208,176))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=32)
    x=str(np.argmax(classes[0]))
    d={'0':'Mild Dementia','1':'Moderate Dementia','2':'No Dementia','3':'Very Mild Dementia'}
    ans=d[x]

    context['ans']=ans

    return render(request,'x.html',context)
