using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;




public class com : MonoBehaviour {


    // Use this for initialization
    public void Start () {


    }
	
	// Update is called once per frame
	void Update () {
		
	}

    void OnCollisionEnter(Collision collision)
    {
        Thread threada = new Thread(new ThreadStart(sendsomething));
        threada.Name = "A Thread";
        threada.Start();

    }

    void sendsomething()
    {

            ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
            using (RequestSocket client = new RequestSocket())
            {
                client.Connect("tcp://localhost:5555");

                for (int i = 0; i < 1; i++)
                {
                    Debug.Log("Sending Hello");
                    client.SendFrame("Hello");

                    var message = client.ReceiveFrameString();
                    Debug.Log("Received" + client.ReceiveFrameString());
                }
            }
            NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yetNetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet
        
        
    }



}
