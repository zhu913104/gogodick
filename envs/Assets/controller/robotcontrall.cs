using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class robotcontrall : MonoBehaviour {


    public float move_speed = 10.0f;
    public float jump_speed = 0.5f;
    Vector2 mouseLook ;
    Vector2 smoothV;
    public float senstivity = 0.5f;
    public float smoothing = 10.0f;
    GameObject character;
    // Use this for initialization
    void Start () {
        //Cursor.lockState = CursorLockMode.Locked;
        character = this.transform.gameObject;
        mouseLook.x = this.GetComponent<Transform>().eulerAngles.y;
        
    }
	
	// Update is called once per frame
	void Update () {
        if (Input.GetKeyDown(KeyCode.R))
        {
            Application.LoadLevel(Application.loadedLevel);
        }


        float translation = Input.GetAxis("Vertical") * move_speed;
        translation *= Time.deltaTime;
        float jump = Input.GetAxis("Jump") * jump_speed;
        transform.Translate(0, jump, translation);

        if (Input.GetKeyDown("escape"))
        {
            Application.Quit();
        }


        var md = new Vector2(Input.GetAxisRaw("Horizontal"), 0);

        md = Vector2.Scale(md, new Vector2(senstivity * smoothing, senstivity * smoothing));
        smoothV.x = Mathf.Lerp(smoothV.x, md.x, 1f / smoothing);
        mouseLook += smoothV;

        //transform.localRotation = Quaternion.AngleAxis(-mouseLook.y, Vector3.right);
        character.transform.localRotation = Quaternion.AngleAxis(mouseLook.x, character.transform.up);

    }
}
