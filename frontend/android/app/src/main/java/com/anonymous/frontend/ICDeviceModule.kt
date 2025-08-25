package com.anonymous.frontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import cn.icomon.icdevicemanager.ICDeviceManager

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val CODE_REQUEST_PERMISSION = 1001
        const val TAG = "ICDeviceModule"
    }

    init {
        // Initialize the SDK
        val manager = ICDeviceManager.shared()
        Log.d(TAG, "ICDeviceManager initialized")
    }

    private val scanDelegate = object : ICScanDeviceDelegate {
        override fun onDeviceFound(device: ICDevice) {
            Log.d(TAG, "Device found: ${device.macAddress}")
            // Send to JS
            val params = mapOf("mac" to device.macAddress, "name" to device.name)
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onDeviceFound", params)
        }

        override fun onScanFinished() {
            Log.d(TAG, "Scan finished")
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                .emit("onScanFinished", null)
        }
    }

    override fun getName(): String = "ICDeviceModule"

    private fun checkBLEConnectionPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.checkSelfPermission(
                reactContext.currentActivity!!,
                Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        reactContext.currentActivity!!,
                        Manifest.permission.BLUETOOTH_SCAN
                    ) == PackageManager.PERMISSION_GRANTED
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.checkSelfPermission(
                reactContext.currentActivity!!,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        reactContext.currentActivity!!,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    ) == PackageManager.PERMISSION_GRANTED
        } else true
    }

    private fun requestBLEConnectionPermission() {
        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.BLUETOOTH_SCAN)
        } else {
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
        }
        ActivityCompat.requestPermissions(
            reactContext.currentActivity!!,
            permissions,
            CODE_REQUEST_PERMISSION
        )
    }

    @ReactMethod
    fun startScan() {
        if (!checkBLEConnectionPermission()) {
            requestBLEConnectionPermission()
            return
        }

        // Start scan
        ICDeviceManager.shared().scanDevice(scanDelegate)
        Log.d(TAG, "Scan started")
    }

    @ReactMethod
    fun stopScan() {
        ICDeviceManager.shared().stopScan()
        Log.d(TAG, "Scan stopped")
    }

    @ReactMethod
    fun addDevice() {
        Log.d(TAG, "addDevice called")
    }
}
