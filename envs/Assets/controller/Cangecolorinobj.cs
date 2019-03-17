using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;

public class Cangecolorinobj : MonoBehaviour
{
    private int scenesMax = 5; 
    private GameObject UIcolor;
    private bool Touchdown;

    // Start is called before the first frame update
    void Start()
    {
        Touchdown = false;
        UIcolor = GameObject.FindWithTag("color");
        
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.D) && Input.GetKey(KeyCode.W) && Touchdown)
        {
            UIcolor.GetComponent<Image>().color = new Color32(127, 127, 127, 255);

        }
        else if (Input.GetKey(KeyCode.A) && Input.GetKey(KeyCode.W) && Touchdown)
        {
            UIcolor.GetComponent<Image>().color = new Color32(127, 127, 127, 255);
        }

        else if (Input.GetKey(KeyCode.W) && Touchdown)
        {

            UIcolor.GetComponent<Image>().color = new Color32(255, 255, 255, 255);

        }



    }
    void OnCollisionEnter(Collision collision)
    {
        Touchdown = true;
        Debug.Log("INNN");
        if (collision.gameObject.tag == "baba")
        {
            UIcolor.GetComponent<Image>().color = new Color32(0, 0, 0, 255);
            Debug.Log(UIcolor.GetComponent<Image>().color);
            Invoke("loadScene", 1.5f);
            Touchdown = false;

        }


    }
    void loadScene()
    {
        
        UnityEngine.SceneManagement.SceneManager.LoadScene(UnityEngine.Random.Range(0, scenesMax));
    }
}
