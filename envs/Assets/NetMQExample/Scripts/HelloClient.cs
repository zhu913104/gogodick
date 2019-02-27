using UnityEngine;

public class HelloClient : MonoBehaviour
{
    private HelloRequester _helloRequester;

    private void Start()
    {
        _helloRequester = new HelloRequester();
        //_helloRequester.Start();
    }

    private void OnDestroy()
    {
        _helloRequester.Stop();
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("XXX"))
            _helloRequester.Start();

    }

    void OnCollisionExit(Collision collisionInfo)
    {
        if (collisionInfo.gameObject.CompareTag("XXX"))
            _helloRequester.Stop();
    }



}