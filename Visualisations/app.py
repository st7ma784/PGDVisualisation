import torch
import numpy as np
from flask import request, render_template,Flask
import flask
import os 

app = Flask(__name__,template_folder='.')
loss=torch.nn.CrossEntropyLoss(reduction="none")

@app.route("/") 
def index():
    return render_template("index.html")


@torch.no_grad()
@app.route('/demo/data', methods=['GET','POST'])
async def getS():
    data=request.get_json()
    # JS report is { 'label_x': x, 'label_y': y,"pred_x":pred_x,"pred_y":pred_y, "target_x":target_x, "target_y":target_y, "norm":isNorm, "width":  window.innerWidth, "height": window.innerHeight};

    wh=torch.tensor([[data['width'],data['height']]])
    label_x=[float(x[:-2]) for x in filter(lambda a: a != '',data['label_x'])]
    label_y=[float(y[:-2]) for y in filter(lambda a: a != '',data['label_y'])]
    pred_x=[float(x[:-2]) for x in filter(lambda a: a != '',data['pred_x'])]
    pred_y=[float(y[:-2]) for y in filter(lambda a: a != '',data['pred_y'])]
    target_x=data['target_x']
    target_y=data['target_y']
    norm=data['norm']
    label_xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(label_x,label_y)])-(wh/2)
    pred_xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(pred_x,pred_y)])-(wh/2)
    target_xys=torch.tensor([[target_x,target_y]],requires_grad=False)-(wh/2)
    theta = torch.linspace(0, 2 * np.pi, 40)
    radius = 0.15
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    circle = torch.stack([cos_theta, sin_theta], dim=-1)*radius 
    circle=circle.unsqueeze(0)
    pred_xys=pred_xys.unsqueeze(1)
    #pred is shape (B,1,2) 
    #circle is shape (1,40,2)
    #we want to add the circle to each pred
    #so we get a shape (B,40,2)
    pred_xys=pred_xys+circle
    #shape (B,40,2)
    pred_xys=pred_xys.view(-1,2)
    
    if norm:
        label_xys=label_xys/torch.norm(label_xys,dim=-1,keepdim=True)
        pred_xys=pred_xys/torch.norm(pred_xys,dim=-1,keepdim=True)
        target_xys=target_xys/torch.norm(target_xys,dim=-1,keepdim=True)
        pred_xys=pred_xys/torch.norm(pred_xys,dim=-1,keepdim=True)

    all_targets=torch.cat([target_xys,label_xys])
    #the predictions are the output of an encoder, we want to draw 40 points around each. 
    Matrix=pred_xys@all_targets.T
    #shape is B*40, labels+1
    labels=torch.zeros(Matrix.shape[0])
    values=loss(Matrix,labels,dtype=torch.long)
    #convert values to colour values between 0 and 1
    values=torch.softmax(values,dim=0)
    x=pred_xys[:,0]
    y=pred_xys[:,1]
    

    return {"x":x.cpu().tolist(),"y":y.cpu().tolist(),"values":values.cpu().tolist()}



if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000 )



