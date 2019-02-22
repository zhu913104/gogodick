using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class robotcontrall : MonoBehaviour {


    public float move_speed = 10.0f;
    public float jump_speed = 0.5f;
    Vector2 mouseLook;
    Vector2 smoothV;
    public float senstivity = 0.5f;
    public float smoothing = 10.0f;
    GameObject character;
    // Use this for initialization
    void Start () {
        Cursor.lockState = CursorLockMode.Locked;
        character = this.transform.gameObject;
    }
	
	// Update is called once per frame
	void Update () {
        float translation = Input.GetAxis("Vertical") * move_speed;
        translation *= Time.deltaTime;
        float jump = Input.GetAxis("Jump") * jump_speed;
        transform.Translate(0, jump, translation);

        if (Input.GetKeyDown("escape"))
        {
            Cursor.lockState = CursorLockMode.None;
        }


        var md = new Vector2(Input.GetAxisRaw("Horizontal"), Input.GetAxisRaw("Mouse Y"));

        md = Vector2.Scale(md, new Vector2(senstivity * smoothing, senstivity * smoothing));
        smoothV.x = Mathf.Lerp(smoothV.x, md.x, 1f / smoothing);
        smoothV.y = 0;
        mouseLook += smoothV;

        transform.localRotation = Quaternion.AngleAxis(-mouseLook.y, Vector3.right);
        character.transform.localRotation = Quaternion.AngleAxis(mouseLook.x, character.transform.up);

    }
}
