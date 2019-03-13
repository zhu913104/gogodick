using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using System.Threading;

public class NewBehaviourScript : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {

        if (Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.W)) 
        {
            Thread threada = new Thread(new ThreadStart(reword_0));
            threada.Name = "A Thread";
            threada.Start();
            
        }
        else if (Input.GetKey(KeyCode.A) && Input.GetKey(KeyCode.W))
        {
            Thread threada = new Thread(new ThreadStart(reword_0));
            threada.Name = "A Thread";
            threada.Start();

        }

        else if(Input.GetKey(KeyCode.W))
        {
            Thread threada = new Thread(new ThreadStart(reword_1));
            threada.Name = "A Thread";
            threada.Start();

        }
    }


    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "baba")
        {
            Thread threada = new Thread(new ThreadStart(reword__1));
            threada.Name = "A Thread";
            threada.Start();

            Application.LoadLevel(Application.loadedLevel);
        }
    }
    void reword_1()
    {

        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");

            for (int i = 0; i < 1; i++)
            {
                Debug.Log("Sending :" + "1");
                client.SendFrame("1");

                var message = client.ReceiveFrameString();
                Debug.Log("Received :" + client.ReceiveFrameString());
            }
        }
        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yetNetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet

    }
    void reword_0()
    {

        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");

            for (int i = 0; i < 1; i++)
            {
                Debug.Log("Sending :" + "0");
                client.SendFrame("0");

                var message = client.ReceiveFrameString();
                Debug.Log("Received :" + client.ReceiveFrameString());
            }
        }
        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yetNetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet

    }
    void reword__1()
    {

        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");

            for (int i = 0; i < 1; i++)
            {
                Debug.Log("Sending :" + "-1");
                client.SendFrame("-1");

                var message = client.ReceiveFrameString();
                Debug.Log("Received :" + client.ReceiveFrameString());
            }
        }
        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yetNetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet

    }
}
