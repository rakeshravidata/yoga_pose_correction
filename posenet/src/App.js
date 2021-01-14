// 1. Install dependencies DONE
// 2. Import dependencies DONE
// 3. Setup webcam and canvas DONE
// 4. Define references to those DONE
// 5. Load posenet DONE
// 6. Detect function DONE
// 7. Drawing utilities from tensorflow DONE
// 8. Draw functions DONE

import React, { useRef } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import * as Math from "mathjs"
import { drawKeypoints, drawSkeleton } from "./utilities";


function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);


  //  Load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 640, height: 480 },
      multiplier: 0.75
    });
    //
    setInterval(() => {
      detect(net);
    }, 1000);
  };

  const BlobToImageData = function(blob){
    let blobUrl = URL.createObjectURL(blob);
    return new Promise((resolve, reject) => {
                let img = new Image();
                img.onload = () => resolve(img);
                img.onerror = err => reject(err);
                img.src = blobUrl;
            }).then(img => {
                URL.revokeObjectURL(blobUrl);
                let [w,h] = [img.width,img.height]
                let aspectRatio = w/h
                let canvas = document.createElement("canvas");
                canvas.width = w;
                canvas.height = h;
                let ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0);
                return ctx.getImageData(0, 0, w, h);    // some browsers synchronously decode image here
            })
}

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const imageSrc = webcamRef.current.getScreenshot({width: 640, height: 480});
      const blob = await fetch(imageSrc).then((res) => res.blob());
      const promise = BlobToImageData(blob)

      // console.log(image_data)
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      //variables for model
      const imageScaleFactor = 0.50;
      const flipHorizontal = false;
      const outputStride = 16;

      // Make Detections
      promise.then((data) => {
      const image_data =  data;
      const pose =  net.estimateSinglePose(video,imageScaleFactor, flipHorizontal, outputStride);
      pose.then((data) => {
        var pose = data
        var arr_x = [];
        var arr_y = [];
        var arr_score = [];
        var avg_score = pose["score"]
        // console.log(avg_score)
        if (avg_score > 0.3){
        for (var val in pose["keypoints"] )  
        {
          arr_x.push( pose["keypoints"][val]["position"]["x"]);
          arr_y.push( pose["keypoints"][val]["position"]["y"]);
          arr_score.push( pose["keypoints"][val]["score"]);
          }
        
        
        var nrm_arr_x  = nrm_coor(arr_x)
        var nrm_arr_y = nrm_coor(arr_y)
        var feedback_list = compare_with_baseline(nrm_arr_x, nrm_arr_y)
        drawCanvas(pose, video, videoWidth, videoHeight, canvasRef, feedback_list);

        
      }
      else {
        drawCanvas_low_conf(video, videoWidth, videoHeight,canvasRef);
        console.log("MOVE BACK")
      }
    })
      
      });

  };

};

  

  const compare_with_baseline = (nrm_arr_x, nrm_arr_y) => {
    var mean_x = [0.9174949448906107, 0.9200841600464503, 0.9222179140956414, 0.9234289896861129, 0.9111519378949061, 0.8727369598052188, 0.8699234026503035, 0.8540390027005511, 0.8520158073034846, 0.8796380642323703, 0.860208891785093, 0.7757548098222566, 0.7614857006985549, 0.7112098003918734, 0.7192274828080291, 0.6683855422355198, 0.6668194625054322]
    var mean_y = [0.7221777808584776, 0.7238162611602144, 0.7376973400405193, 0.69591916408011, 0.7326812808210252, 0.6926902978153417, 0.7460935231473801, 0.6697452108507556, 0.7985159251309434, 0.6900659248132935, 0.7631413589699575, 0.7199172245794602, 0.7649172935380986, 0.7604420821889375, 0.768029913254374, 0.7540867439004668, 0.7436509951449335]
    var std_x = [0.054950125999660496, 0.052210451863514244, 0.05428489328845133, 0.04987098568598721, 0.07898320266742635, 0.08492647249657446, 0.08897458884992848, 0.10949433288120242, 0.12308454247391735, 0.08362703253714276, 0.12684954948052438, 0.15131598139987945, 0.1490303013133029, 0.18784490838877363, 0.16171670188177642, 0.2088128021991637, 0.1967713268181371]
    var std_y = [0.1620702247749683, 0.16182841097600287, 0.1535220575470839, 0.18003407128170448, 0.15928773912419836, 0.1845718794996506, 0.1491668091101181, 0.2170058357185045, 0.13351609094894207, 0.2048618504133284, 0.15453503940721486, 0.17593012817876325, 0.14751073461658384, 0.15394068000392142, 0.1459478469510186, 0.1374980546202483, 0.1378052601659767]
    var parts = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    var failed_parts = [];
    var feedback_list = []; 
    for (var val in nrm_arr_x )  
    {
      var x = nrm_arr_x[val]
      var y = nrm_arr_y[val]
      var x_mean = mean_x[val]
      var y_mean = mean_y[val]
      var x_std = std_x[val]
      var y_std = std_y[val]
      var x_percDiff = 100 * Math.abs( (x - x_mean) / ( x_mean));
      var y_percDiff = 100 * Math.abs( (y - y_mean) / ( y_mean));
      var x_pass = 0
      var y_pass = 0
      var part_pass = 1
      if ((x_percDiff < 20) && (Math.abs(x-x_mean) < x_std)){
        x_pass = 1
      }
      if ((y_percDiff < 20) && (Math.abs(y-y_mean) < y_std)){
        y_pass = 1
      }
      if ((x_pass === 0) && (y_pass === 0)){
        part_pass = 0
      } 
      if ((part_pass === 0) && (val > 4)){
        var part_name = parts[val]
        failed_parts.push( part_name);
        if ((y-y_mean) > 0){
          var feedback = "Please Lower your " + part_name + " slightly!"
          feedback_list.push(feedback)
        }
        // if ((y-y_mean) < 0){
        //   var feedback = "Please Raise your " + part_name + " slightly!"
        //   feedback_list.push(feedback)
        // }
      }
      }
    return feedback_list
    
  };
  const nrm_coor = (arr) => {
    const buffer = 50 
    const max = Math.max(arr) + buffer
    const min = Math.min(arr) - buffer
    var nrm_arr = [];
    for (var val in arr )  
    {
      var nrm = (arr[val] - min)/(max-min)
      nrm_arr.push( nrm);
      }
    return nrm_arr
  };

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas, feedback_list) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;
    drawKeypoints(pose["keypoints"], 0.3, ctx);
    drawSkeleton(pose["keypoints"], 0.3, ctx);
    console.log(feedback_list.length)
    if (feedback_list.length >= 2){
    for (var ele in feedback_list )  
    {
      var feedback = feedback_list[ele]
      ctx.font = '25px serif';
      ctx.fillText(feedback, 0, 180 - 30*ele);
      }
    }
    else {
      ctx.font = '25px serif';
      ctx.fillText('PERFECT', 0, 180 - 30);
    }

  };

  const drawCanvas_low_conf = (video, videoWidth, videoHeight,canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;
    ctx.textAlign = "start";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = "#ff0000";
    ctx.font = "bold 30px verdana, sans-serif";
    ctx.fillText('PLEASE MOVE BACK', videoWidth/2, videoHeight/2);
  };

  runPosenet();

  return (
    <div className = "App">
      <h1>Yoga Pose Corrector</h1>

      <div className="App">
        <header className="App-header">
          <Webcam
            ref={webcamRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zindex: 9,
              width: 1280,
              height: 960,
            }}
          />
          
          <canvas
            ref={canvasRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zindex: 9,
              width: 1280,
              height: 960,
            }}
          />
        
        </header>
      </div>
      </div>


    

  );
}

export default App;