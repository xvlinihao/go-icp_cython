from py_goicp import GoICP, POINT3D, ROTNODE, TRANSNODE;
import numpy as np;
import time;
import argparse
import os

def loadPointCloud(filename):
    pcloud = np.load(filename);
    plist = pcloud.tolist();
    p3dlist = [];
    for x,y,z in plist:
        pt = POINT3D(x,y,z);
        p3dlist.append(pt);
    return pcloud.shape[0], p3dlist, pcloud;
def main():
    parser = argparse.ArgumentParser(description="input file")
    parser.add_argument("--src",type=str,default='',metavar='N',
    help="source file")
    parser.add_argument('--tgt',type=str,default='',metavar='N',
    help='target file')
    parser.add_argument('--name',type=str,default='',metavar='N',
    help='the name of the registration result')

    args=parser.parse_args()
    
    goicp = GoICP();
    rNode = ROTNODE();
    tNode = TRANSNODE();
 
    rNode.a = -3.1416;
    rNode.b = -3.1416;
    rNode.c = -3.1416;
    rNode.w = 6.2832;
 
    tNode.x = -0.5;
    tNode.y = -0.5;
    tNode.z = -0.5;
    tNode.w = 1.0;

    goicp.MSEThresh = 0.001;
    goicp.trimFraction = 0.0;
  
    if(goicp.trimFraction < 0.001):
        goicp.doTrim = False;

    a_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(0.5, 1.0, 0.0), POINT3D(1.0, 0.0, 0.0)];
    b_points = [POINT3D(0.0, 0.0, 0.0), POINT3D(1.0, 0.5, 0.0), POINT3D(1.0, -0.5, 0.0)];

    Nm, a_points, np_a_points = loadPointCloud(args.src);
    Nd, b_points, np_b_points = loadPointCloud(args.tgt);

    goicp.loadModelAndData(Nm, a_points, Nd, b_points);
#LESS DT Size = LESS TIME CONSUMPTION = HIGHER ERROR
    goicp.setDTSizeAndFactor(300, 2.0);
    goicp.setInitNodeRot(rNode);
    goicp.setInitNodeTrans(tNode);

    start = time.time();
    print("Building Distance Transform...");
    goicp.BuildDT();
    print("REGISTERING....");
    goicp.Register();
    end = time.time();
    print('TOTAL TIME : ', (end-start));
    optR = np.array(goicp.optimalRotation());
    optT = goicp.optimalTranslation();
    optT.append(1.0);
    optT = np.array(optT);

    transform = np.empty((4,4));
    transform[:3, :3] = optR;
    transform[:,3] = optT;

    print(np_b_points.shape, np.ones((Nd, 1)).shape);

#Now transform the data mesh to fit the model mesh
    transform_model_points = (transform.dot(np.hstack((np_b_points, np.ones((Nd, 1)))).T)).T;
    transform_model_points = transform_model_points[:,:3];
    if not os.path.exists('indermidiate_data'):
        os.makedirs('indermidiate_data')
    #PLY_FILE_HEADER = "ply\nformat ascii 1.0\ncomment PYTHON generated\nelement vertex %s\nproperty float x\nproperty float y\nproperty float z\nend_header"%(Nd);
    np.save('indermidiate_data/{}.npy'.format(args.name),transform_model_points)
    #np.savetxt('/content/go-icp_cython/tests/data_bunny_transformed.ply', transform_model_points, header = PLY_FILE_HEADER, comments='');

    print(optR);
    print(optT);
    print(transform);

if __name__ == "__main__":
    main()




