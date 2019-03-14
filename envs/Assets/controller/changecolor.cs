using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class changecolor : MonoBehaviour
{

    public Image imagecolor;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.W))
        {
            imagecolor.color = new Color32(127, 127, 127, 255);
        }
        else if (Input.GetKey(KeyCode.A) && Input.GetKey(KeyCode.W))
        {
            imagecolor.color = new Color32(127, 127, 127, 255);

        }

        else if (Input.GetKey(KeyCode.W))
        {
            imagecolor.color = new Color32(255, 255, 255, 255);
        }

    }
    void OnCollisionEnter(Collision collision)
    {


        if (collision.gameObject.tag == "baba")
        {
            //Thread threada = new Thread(new ThreadStart(reword__1));
            //threada.Name = "A Thread";
            //threada.Start();
            //Debug.Log("-1");


            UnityEngine.SceneManagement.SceneManager.LoadScene("2");


        }
    }
}
