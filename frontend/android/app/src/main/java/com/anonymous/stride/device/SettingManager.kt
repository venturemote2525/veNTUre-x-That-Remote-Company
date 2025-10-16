package com.anonymous.stride.device

import android.util.Log
import com.facebook.react.bridge.*
import cn.icomon.icdevicemanager.ICDeviceManager
import cn.icomon.icdevicemanager.ICDeviceManagerSettingManager
import cn.icomon.icdevicemanager.ICDeviceManagerSettingManager.ICSettingCallback
import cn.icomon.icdevicemanager.model.device.ICDevice
import cn.icomon.icdevicemanager.model.device.ICUserInfo
import cn.icomon.icdevicemanager.model.other.ICConstant

class SettingManager(private val reactContext: ReactApplicationContext, private val scanManager: ScanManager) {

    companion object {
        const val TAG = "ICSettingManager"
    }

    private val deviceManager = scanManager.getDeviceManager()
    private val settingManager = deviceManager.getSettingManager()

    private fun getDevice(mac: String): ICDevice? {
        return scanManager.getConnectedDevice(mac)
    }

    // -------------------- COMMON SETTINGS --------------------

    fun setUserInfo(mac: String, userInfo: ICUserInfo, promise: Promise) {
        val device = getDevice(mac)
        Log.d(TAG, "Device: $device")
        if (device == null) {
            promise.reject("DEVICE_NOT_FOUND", "Device not connected")
            return
        }
        Log.d(TAG, "setUserInfo for $mac. User: $userInfo")
        settingManager.setUserInfo(device, userInfo, object : ICDeviceManagerSettingManager.ICSettingCallback {
            override fun onCallBack(code: ICConstant.ICSettingCallBackCode) {
                when (code) {
                    ICConstant.ICSettingCallBackCode.ICSettingCallBackCodeSuccess -> promise.resolve(true)
                    else -> promise.reject("SET_USER_FAIL", "Failed with code $code")
                }
            }
        })
    }
}
