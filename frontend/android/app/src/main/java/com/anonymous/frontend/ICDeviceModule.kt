package com.anonymous.frontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.anonymous.frontend.device.PermissionManager
import com.anonymous.frontend.device.ScanManager

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val permissionManager = PermissionManager(reactContext)
    private val scanManager = ScanManager(reactContext)

    override fun getName(): String = "ICDeviceModule"

    @ReactMethod
    fun initializeSDK() {
        if (!permissionManager.hasBLEPermission()) {
            permissionManager.requestBLEPermission()
            return
        }
        scanManager.initSDK()
    }

    @ReactMethod
    fun startScan() {
        if (!permissionManager.hasBLEPermission()) {
            permissionManager.requestBLEPermission()
            return
        }
        scanManager.startScan()
    }

    @ReactMethod
    fun stopScan() = scanManager.stopScan()

    @ReactMethod
    fun addDevice(mac: String, name: String) = scanManager.addDevice(mac, name)

    @ReactMethod
    fun removeDevice(mac: String) = scanManager.removeDevice(mac)

    @ReactMethod
    fun getScannedDevices() = emitToJS(
        "onScannedDevices",
        Arguments.createArray().apply {
            scanManager.getScannedDevices().forEach { device ->
                pushMap(Arguments.createMap().apply {
                    putString("mac", device.mac)
                    putString("name", device.name)
                })
            }
        }
    )

    @ReactMethod
    fun getConnectedDevices() = emitToJS(
        "onConnectedDevices",
        Arguments.createArray().apply {
            scanManager.getConnectedDevices().forEach { device ->
                pushMap(Arguments.createMap().apply {
                    putString("mac", device.mac)
                    putString("name", device.name)
                })
            }
        }
    )

    private fun emitToJS(event: String, params: Any?) =
        reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
}
