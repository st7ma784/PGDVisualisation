import torch
import numpy as np
from flask import request, render_template,Flask,jsonify
import flask
import os 

app = Flask(__name__,template_folder='.')
loss=torch.nn.CrossEntropyLoss(reduction="none")

@app.route("/") 
def index():
    return render_template("index.html")


@torch.no_grad()
@app.route('/data', methods=['GET','POST'])
async def getS():
    data=request.get_json()
    # print("data is",data)
    # JS report is { 'label_x': x, 'label_y': y,"pred_x":pred_x,"pred_y":pred_y, "target_x":target_x, "target_y":target_y, "norm":isNorm, "width":  window.innerWidth, "height": window.innerHeight};

    wh=torch.tensor([[data['width'],data['height']]])
    label_x=[float(x[:-2]) for x in filter(lambda a: a != '',data['labelx'])]
    label_y=[float(y[:-2]) for y in filter(lambda a: a != '',data['labely'])]
    pred_x=[float(x[:-2]) for x in filter(lambda a: a != '',data['predx'])]
    pred_y=[float(y[:-2]) for y in filter(lambda a: a != '',data['predy'])]
    target_x=float(data['targetx'][:-2])
    target_y=float(data['targety'][:-2])
    norm=data['norm']
    radius=int(data['radius'])/100
    num_points=int(data['numlabels'])
    label_xys=torch.stack([torch.tensor([x,y],requires_grad=False)for x,y in zip(label_x,label_y)])-(wh/2)
    pred_xys=torch.stack([torch.tensor([[x,y]],requires_grad=False)for x,y in zip(pred_x,pred_y)])-(wh/2)
    target_xys=torch.tensor([[target_x,target_y]],requires_grad=False)-(wh/2)

    label_xys=label_xys/wh
    pred_xys=pred_xys/wh
    target_xys=target_xys/wh

    theta = torch.linspace(0, 2 * np.pi, num_points)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    circle = torch.stack([cos_theta, sin_theta], dim=-1)*radius 
    circle=circle.unsqueeze(0)
    
    #pred is shape (B,1,2) 
    #circle is shape (1,40,2)
    #we want to add the circle to each pred
    #so we get a shape (B,40,2)
    pred_xys=pred_xys+circle
    #shape (B,40,2)
    pred_xys=pred_xys.view(-1,2)
    pred_xysa=pred_xys.view(-1,2)
    if norm:
        label_xys=label_xys/torch.norm(label_xys,dim=-1,keepdim=True)
        pred_xys=pred_xys/torch.norm(pred_xys,dim=-1,keepdim=True)
        target_xys=target_xys/torch.norm(target_xys,dim=-1,keepdim=True)
        pred_xys=pred_xys/torch.norm(pred_xys,dim=-1,keepdim=True)

    # print("label_xys",label_xys.shape)
    # print("pred_xys",pred_xys.shape)
    # print("target_xys",target_xys.shape)
    all_targets=torch.cat([target_xys,label_xys],dim=0)
    #the predictions are the output of an encoder, we want to draw 40 points around each. 
    Matrix=pred_xys@all_targets.T
    #shape is B*40, labels+1
    labels=torch.zeros(Matrix.shape[0],dtype=torch.long)
    values=loss(Matrix,labels)
    #convert values to colour values between 0 and 1
    values=values-values.min()
    values=values/values.max()
    # print("values",values)
    # unnorm the xys and return
    pred_xysa=pred_xysa*wh 
    pred_xysa=pred_xysa+(wh/2)
    x=pred_xysa[:,0]
    y=pred_xysa[:,1]
    
    dictionary={"x":x.cpu().tolist(),"y":y.cpu().tolist(),"values":values.cpu().tolist()}
    # print(dictionary)
    #convert this to a json object and return it
    return jsonify(dictionary)


#write pytests
def test_getS():
    with app.test_client() as c:
        response = c.post('/data', json={'label_x': [0,0], 'label_y': [0,0],"pred_x":[0,0],"pred_y":[0,0], "target_x":0, "target_y":0, "norm":False, "width":  100, "height": 100})
        assert response.status_code == 200

if __name__ == "__main__":

    app.run(host="localhost", port=5000 )



