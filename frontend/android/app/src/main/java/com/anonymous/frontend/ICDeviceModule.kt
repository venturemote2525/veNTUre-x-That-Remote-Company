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
import com.anonymous.frontend.device.SettingManager
import cn.icomon.icdevicemanager.model.device.ICUserInfo
import cn.icomon.icdevicemanager.model.other.ICConstant

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private val permissionManager = PermissionManager(reactContext)
    private val scanManager = ScanManager(reactContext)
    private val settingManager = SettingManager(reactContext, scanManager)

    override fun getName(): String = "ICDeviceModule"

    companion object {
        const val TAG = "ICDeviceModule"
    }

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
    fun getBleState(promise: Promise) {
        try {
            val adapter = android.bluetooth.BluetoothAdapter.getDefaultAdapter()
            if (adapter == null) {
                // Device doesn't support Bluetooth
                val result = Arguments.createMap()
                result.putString("state", "Unsupported")
                result.putBoolean("enabled", false)
                promise.resolve(result)
                return
            }

            val enabled = adapter.isEnabled
            val state = if (enabled) "ICBleStatePoweredOn" else "ICBleStatePoweredOff"

            val result = Arguments.createMap()
            result.putString("state", state)
            result.putBoolean("enabled", enabled)

            promise.resolve(result)
        } catch (e: Exception) {
            promise.reject("BLE_STATE_ERROR", e.message, e)
        }
    }

    // -------------------- SETTING METHODS --------------------

    @ReactMethod
    fun setUserInfo(mac: String, userInfoMap: ReadableMap, promise: Promise) {
        Log.d(TAG, "Set user info")
        val userInfo = ICUserInfo()
        userInfo.nickName = userInfoMap.getString("name") ?: ""
        userInfo.age = userInfoMap.getInt("age")
        userInfo.height = userInfoMap.getInt("height")
        val genderStr = userInfoMap.getString("gender") ?: "MALE"
        userInfo.sex = when (genderStr.uppercase()) {
            "FEMALE" -> ICConstant.ICSexType.ICSexTypeFemal
            "MALE" -> ICConstant.ICSexType.ICSexTypeMale
            else -> ICConstant.ICSexType.ICSexTypeMale
        }
        Log.d(TAG, "Set current user info: $userInfo")
        settingManager.setUserInfo(mac, userInfo, promise)
    }

    @ReactMethod
    fun updateUserInfo(userInfoMap: ReadableMap) = scanManager.updateUserInfo(userInfoMap)

    @ReactMethod
    fun updateUserInfo_W(mac: String, userInfoMap: ReadableMap, promise: Promise) {
        Log.d(TAG, "Set user info")
        val userInfo = ICUserInfo()
        userInfo.nickName = userInfoMap.getString("name") ?: ""
        userInfo.age = userInfoMap.getInt("age")
        userInfo.height = userInfoMap.getInt("height")
        val genderStr = userInfoMap.getString("gender") ?: "MALE"
        userInfo.sex = when (genderStr.uppercase()) {
            "FEMALE" -> ICConstant.ICSexType.ICSexTypeFemal
            "MALE" -> ICConstant.ICSexType.ICSexTypeMale
            else -> ICConstant.ICSexType.ICSexTypeMale
        }
        Log.d(TAG, "Set current user info: $userInfo")
        settingManager.updateUserInfo_W(mac, userInfo, promise)
    }

    @ReactMethod
    fun getUserList_W(mac: String, promise: Promise) {
        Log.d(TAG, "Get user list W")
        settingManager.getUserList_W(mac, promise)
    }

    @ReactMethod
    fun setUserList(userInfoMap: ReadableMap) {
        Log.d(TAG, "Set user list")
        scanManager.setUserList(userInfoMap)
    }
}