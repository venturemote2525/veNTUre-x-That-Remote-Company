package com.anonymous.frontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import com.anonymous.frontend.device.PermissionManager
import com.anonymous.frontend.device.ScanManager

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val permissionManager = PermissionManager(reactContext)
    private val scanManager = ScanManager(reactContext)

    override fun getName(): String = "ICDeviceModule"

    @ReactMethod
    fun initializeSDK(promise: Promise) = scanManager.initSDK(promise)

    @ReactMethod
    fun isSDKInitialized(promise: Promise) {
        promise.resolve(scanManager.isSDKInitialized())
    }

    @ReactMethod
    fun startScan(promise: Promise) = scanManager.startScan(promise)

    @ReactMethod
    fun stopScan(promise: Promise) = scanManager.stopScan(promise)

    @ReactMethod
    fun connectDevice(mac: String, promise: Promise) = scanManager.connectDevice(mac, promise)

    @ReactMethod
    fun disconnectDevice(mac: String, promise: Promise) = scanManager.disconnectDevice(mac, promise)

    @ReactMethod
    fun getScannedDevices(promise: Promise) = scanManager.getScannedDevices(promise)

    @ReactMethod
    fun getConnectedDevices(promise: Promise) = scanManager.getConnectedDevices(promise)

    @ReactMethod
    fun clearScannedDevices(promise: Promise) = scanManager.clearScannedDevices(promise)

    @ReactMethod
    fun addListener(eventName: String) {
        scanManager.incrementListenerCount(eventName)
    }

    @ReactMethod
    fun removeListeners(count: Double) {
        scanManager.decrementListenerCount(count.toInt())
    }

    @ReactMethod
    fun isDeviceConnected(macAddress: String, promise: Promise) {
        promise.resolve(scanManager.isDeviceConnected(macAddress))
    }

    @ReactMethod
    fun isScanning(promise: Promise) {
        promise.resolve(scanManager.isScanning())
    }

    @ReactMethod
    fun getBodyFatAlgorithmsManager(promise: Promise) {
        try {
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }

    @ReactMethod
    fun addDevice(mac: String, name: String) = scanManager.addDevice(mac, name)

    @ReactMethod
    fun removeDevice(mac: String) = scanManager.removeDevice(mac)
}