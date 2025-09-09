package com.anonymous.frontend.device

import android.util.Log
import android.bluetooth.BluetoothAdapter
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import com.facebook.react.bridge.*
import com.facebook.react.modules.core.DeviceEventManagerModule
import cn.icomon.icdevicemanager.ICDeviceManager
import cn.icomon.icdevicemanager.ICDeviceManagerDelegate
import cn.icomon.icdevicemanager.callback.ICScanDeviceDelegate
import cn.icomon.icdevicemanager.model.device.ICDevice
import cn.icomon.icdevicemanager.model.device.ICScanDeviceInfo
import cn.icomon.icdevicemanager.model.device.ICUserInfo
import cn.icomon.icdevicemanager.model.device.ICDeviceInfo
import cn.icomon.icdevicemanager.model.other.ICConstant
import cn.icomon.icdevicemanager.model.other.ICDeviceManagerConfig
import cn.icomon.icdevicemanager.model.data.ICWeightData
import cn.icomon.icdevicemanager.model.data.ICWeightCenterData
import cn.icomon.icdevicemanager.model.data.ICKitchenScaleData
import cn.icomon.icdevicemanager.model.data.ICCoordData
import cn.icomon.icdevicemanager.model.data.ICWeightHistoryData
import cn.icomon.icdevicemanager.model.data.ICRulerData
import cn.icomon.icdevicemanager.model.data.ICSkipData

class ScanManager(private val reactContext: ReactApplicationContext) {

    companion object {
        const val TAG = "ICScanManager"
    }

    private val deviceManager: ICDeviceManager = ICDeviceManager.shared()
    fun getDeviceManager(): ICDeviceManager {
        return deviceManager
    }

    private val scannedDevices = mutableMapOf<String, ICDevice>()
    private val connectedDevices = mutableMapOf<String, ICDevice>()
    private var listenerCount = 0
    private var scanning = false
    private var sdkInitialized = false
    private var receiverRegistered = false

    fun isSDKInitialized() = sdkInitialized
    fun isScanning() = scanning

    // -------------------- LISTENERS --------------------

    fun incrementListenerCount(eventName: String) {
        listenerCount++
        Log.d(TAG, "Added listener for $eventName, total: $listenerCount")
    }

    fun decrementListenerCount(count: Int) {
        listenerCount = (listenerCount - count).coerceAtLeast(0)
        Log.d(TAG, "Removed ${count.toInt()} listeners, remaining: $listenerCount")
    }

    fun hasListeners(): Boolean = listenerCount > 0

    // -------------------- INIT SDK --------------------

    fun initSDK(userInfoMap: ReadableMap, promise: Promise) {
        try {
            val config = ICDeviceManagerConfig()
            config.context = reactContext.applicationContext
            deviceManager.setDelegate(deviceManagerDelegate)

            Log.d(TAG, "Set user info")
            val userInfo = ICUserInfo()
            Log.d(TAG, "Original user info: $userInfo")
            userInfo.nickName = userInfoMap.getString("name") ?: ""
            userInfo.age = userInfoMap.getInt("age")
            userInfo.height = userInfoMap.getInt("height")
            val genderStr = userInfoMap.getString("gender") ?: "MALE"
            userInfo.sex = when (genderStr.uppercase()) {
                "FEMALE" -> ICConstant.ICSexType.ICSexTypeFemal
                "MALE" -> ICConstant.ICSexType.ICSexTypeMale
                else -> ICConstant.ICSexType.ICSexTypeMale
            }
            userInfo.peopleType = ICConstant.ICPeopleType.ICPeopleTypeNormal
            Log.d(TAG, "Set current user info: $userInfo")
            deviceManager.updateUserInfo(userInfo)

            sdkInitialized = true
            deviceManager.initMgrWithConfig(config)
            restoreConnectedDevices()
            Log.d(TAG, "SDK initialized")
            promise.resolve(true)

            if (!receiverRegistered) {
                val filter = IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED)
                reactContext.registerReceiver(bluetoothReceiver, filter)
                receiverRegistered = true
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize SDK: ${e.message}")
            promise.reject("INIT_ERROR", e.message)
        }
    }

    fun updateUserInfo(userInfoMap: ReadableMap) {
        Log.d(TAG, "Set user info")
        val userInfo = ICUserInfo()
        Log.d(TAG, "Original user info: $userInfo")
        userInfo.nickName = userInfoMap.getString("name") ?: ""
        userInfo.age = userInfoMap.getInt("age")
        userInfo.height = userInfoMap.getInt("height")
        val genderStr = userInfoMap.getString("gender") ?: "MALE"
        userInfo.sex = when (genderStr.uppercase()) {
            "FEMALE" -> ICConstant.ICSexType.ICSexTypeFemal
            "MALE" -> ICConstant.ICSexType.ICSexTypeMale
            else -> ICConstant.ICSexType.ICSexTypeMale
        }
        userInfo.peopleType = ICConstant.ICPeopleType.ICPeopleTypeNormal
        Log.d(TAG, "Set current user info: $userInfo")
        deviceManager.updateUserInfo(userInfo)
    }

    // -------------------- SCAN --------------------

    fun startScan(promise: Promise) {
        if (!sdkInitialized) {
            promise.reject("SDK_NOT_INITIALIZED", "SDK not initialized")
            return
        }
        try {
            scannedDevices.clear()
            deviceManager.scanDevice(scanDelegate)
            scanning = true
            Log.d(TAG, "Scan started")
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("SCAN_ERROR", e.message)
        }
    }

    fun stopScan(promise: Promise) {
        try {
            deviceManager.stopScan()
            scanning = false
            Log.d(TAG, "Scan stopped")
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("STOP_SCAN_ERROR", e.message)
        }
    }

    // -------------------- CONNECT / DISCONNECT --------------------

    fun connectDevice(mac: String, promise: Promise) {
        val device = scannedDevices[mac]
        if (device == null) {
            promise.reject("DEVICE_NOT_FOUND", "Device with MAC $mac not found in scanned devices")
            return
        }
        deviceManager.addDevice(device) { d, code ->
            Log.d(TAG, "Connect callback $mac -> $code")
            if (code == ICConstant.ICAddDeviceCallBackCode.ICAddDeviceCallBackCodeSuccess) {
                connectedDevices[mac] = d
                Log.d(TAG, "Connect device: $d")
                promise.resolve(true)
            } else {
                promise.reject("CONNECTION_ERROR", "Failed with code $code")
            }
        }
    }

    fun disconnectDevice(mac: String, promise: Promise) {
        try {
            val device = connectedDevices[mac] ?: scannedDevices[mac]
            if (device == null) {
                promise.reject("DEVICE_NOT_FOUND", "Device with MAC $mac not found")
                return
            }
            deviceManager.removeDevice(device, object : ICConstant.ICRemoveDeviceCallBack {
                override fun onCallBack(device: ICDevice, code: ICConstant.ICRemoveDeviceCallBackCode) {
                    Log.d(TAG, "Device remove callback: ${device.macAddr} -> $code")
                    when (code) {
                        ICConstant.ICRemoveDeviceCallBackCode.ICRemoveDeviceCallBackCodeSuccess -> {
                            connectedDevices.remove(mac)
                            promise.resolve(true)
                        }
                        else -> {
                            // Handle all other cases - log and resolve as success
                            Log.d(TAG, "Device remove result: $code")
                            connectedDevices.remove(mac)
                            promise.resolve(true)
                        }
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error disconnecting from device: ${e.message}")
            promise.reject("DISCONNECTION_ERROR", e.message)
        }
    }

    fun isDeviceConnected(mac: String): Boolean {
        return connectedDevices.containsKey(mac)
    }

    // -------------------- DEVICE LISTS --------------------

    fun getScannedDevices(promise: Promise) {
        val arr = Arguments.createArray()
        scannedDevices.values.forEach { d ->
            arr.pushMap(Arguments.createMap().apply {
                putString("mac", d.macAddr)
                putBoolean("isConnected", connectedDevices.containsKey(d.macAddr))
            })
        }
        promise.resolve(arr)
    }

    fun getConnectedDevices(promise: Promise) {
        val arr = Arguments.createArray()
        connectedDevices.values.forEach { d ->
            arr.pushMap(Arguments.createMap().apply {
                putString("mac", d.macAddr)
                putBoolean("isConnected", true)
            })
        }
        promise.resolve(arr)
    }

    fun clearScannedDevices(promise: Promise) {
        scannedDevices.clear()
        promise.resolve(true)
    }

    // -------------------- DELEGATES --------------------

    private val deviceManagerDelegate = object : ICDeviceManagerDelegate {
        override fun onInitFinish(success: Boolean) {
            sdkInitialized = success
            Log.d(TAG, "SDK init finished: $success")
            emitToJS("onSDKInit", Arguments.createMap().apply { putBoolean("success", success) })
        }

        override fun onBleState(state: ICConstant.ICBleState) {
            Log.d(TAG, "BLE state: $state")
            emitToJS("onBleState", Arguments.createMap().apply { putString("state", state.name) })
        }

        override fun onDeviceConnectionChanged(
            device: ICDevice,
            state: ICConstant.ICDeviceConnectState
        ) {
            Log.d(TAG, "Device ${device.macAddr} connection changed: $state")

            when (state) {
                ICConstant.ICDeviceConnectState.ICDeviceConnectStateConnected -> {
                    connectedDevices[device.macAddr] = device
                    Log.d(TAG, "Device ${device.macAddr} connected")
                    // Save to persistent storage
                    val prefs = reactContext.getSharedPreferences("ConnectedDevices", Context.MODE_PRIVATE)
                    prefs.edit().putString(device.macAddr, device.macAddr).apply()
                }
                ICConstant.ICDeviceConnectState.ICDeviceConnectStateDisconnected -> {
                    connectedDevices.remove(device.macAddr)
                    Log.d(TAG, "Device ${device.macAddr} disconnected")
                }
                else -> {
                    Log.d(TAG, "Device ${device.macAddr} state: $state")
                }
            }

            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putString("state", state.name)
                putBoolean("isConnected", state == ICConstant.ICDeviceConnectState.ICDeviceConnectStateConnected)
            }
            emitToJS("onDeviceConnectionChanged", params)
        }

        override fun onReceiveUpgradePercent(
            device: ICDevice,
            status: ICConstant.ICUpgradeStatus,
            percent: Int
        ) {
            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putString("status", status.name)
                putInt("percent", percent)
            }
            emitToJS("onReceiveUpgradePercent", params)
        }
        
        override fun onReceiveBattery(device: ICDevice, battery: Int, ext: Any?) {
            Log.d(TAG, "Received battery level from ${device.macAddr}: $battery%")
            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putInt("battery", battery)
            }
            emitToJS("onReceiveBattery", params)
        }

        override fun onReceiveDeviceInfo(device: ICDevice, deviceInfo: ICDeviceInfo) {
            Log.d(TAG, "Received device info from ${device.macAddr}")
            val infoMap = Arguments.createMap().apply {
                // Remove specific properties that don't exist in this SDK version
                putString("deviceType", "IC Device")
            }
            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putMap("deviceInfo", infoMap)
            }
            emitToJS("onReceiveDeviceInfo", params)
        }

        override fun onReceiveRSSI(device: ICDevice, rssi: Int) {
            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putInt("rssi", rssi)
            }
            emitToJS("onReceiveRSSI", params)
        }

        override fun onReceiveUserInfo(device: ICDevice, userInfo: ICUserInfo) {
            Log.d(TAG, "Received user info from ${device.macAddr}: $userInfo")

            val userMap = Arguments.createMap().apply {
                putString("nickname", userInfo.nickName)
                putInt("age", userInfo.age)
                putInt("height", userInfo.height)
                putString("sex", userInfo.sex.name)
                putString("peopleType", userInfo.peopleType.name)
            }

            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putMap("userInfo", userMap)
            }

            emitToJS("onReceiveUserInfo", params)
        }

        override fun onReceiveUserInfoList(device: ICDevice, list: List<ICUserInfo>) {
            Log.d(TAG, "Received user list from ${device.macAddr}: $list")
        }
        override fun onReceiveHR(device: ICDevice, hr: Int) {}

        // Weight scale callbacks
        override fun onReceiveWeightData(device: ICDevice, data: ICWeightData) {
            // This method is now handled by onReceiveMeasureStepData
            // Keep this override for interface compliance but delegate to the new method
            handleWeightData(device, data)
        }
        override fun onReceiveWeightCenterData(device: ICDevice, data: ICWeightCenterData) {
            // This method is now handled by onReceiveMeasureStepData
            handleWeightCenterData(device, data)
        }
        override fun onReceiveCoordData(device: ICDevice, data: ICCoordData) {}
        override fun onReceiveMeasureStepData(device: ICDevice, step: ICConstant.ICMeasureStep, data: Any?) {
            Log.d(TAG, "Received measure step data from ${device.macAddr}: $step")

            when (step) {
                ICConstant.ICMeasureStep.ICMeasureStepMeasureWeightData -> {
                    val weightData = data as? ICWeightData
                    if (weightData != null) {
                        handleWeightData(device, weightData)
                    }
                }
                ICConstant.ICMeasureStep.ICMeasureStepMeasureCenterData -> {
                    val centerData = data as? ICWeightCenterData
                    if (centerData != null) {
                        handleWeightCenterData(device, centerData)
                    }
                }
                ICConstant.ICMeasureStep.ICMeasureStepAdcStart -> {
                    Log.d(TAG, "${device.macAddr}: start impedance measurement...")
                    val params = Arguments.createMap().apply {
                        putString("mac", device.macAddr)
                        putString("step", "impedanceStart")
                    }
                    emitToJS("onMeasureStep", params)
                }
                ICConstant.ICMeasureStep.ICMeasureStepAdcResult -> {
                    Log.d(TAG, "${device.macAddr}: impedance measurement complete")
                    val params = Arguments.createMap().apply {
                        putString("mac", device.macAddr)
                        putString("step", "impedanceComplete")
                    }
                    emitToJS("onMeasureStep", params)
                }
                ICConstant.ICMeasureStep.ICMeasureStepHrStart -> {
                    Log.d(TAG, "${device.macAddr}: start heart rate measurement")
                    val params = Arguments.createMap().apply {
                        putString("mac", device.macAddr)
                        putString("step", "heartRateStart")
                    }
                    emitToJS("onMeasureStep", params)
                }
                ICConstant.ICMeasureStep.ICMeasureStepHrResult -> {
                    val hrData = data as? ICWeightData
                    if (hrData != null) {
                        Log.d(TAG, "${device.macAddr}: heart rate measurement complete: ${hrData.hr}")
                        val params = Arguments.createMap().apply {
                            putString("mac", device.macAddr)
                            putString("step", "heartRateComplete")
                            putInt("heartRate", hrData.hr)
                        }
                        emitToJS("onMeasureStep", params)
                    }
                }
                ICConstant.ICMeasureStep.ICMeasureStepMeasureOver -> {
                    val weightData = data as? ICWeightData
                    if (weightData != null) {
                        Log.d(TAG, "${device.macAddr}: measurement complete")
                        weightData.isStabilized = true
                        handleWeightData(device, weightData)
                    }
                }
                else -> {
                    Log.d(TAG, "${device.macAddr}: unknown measure step: $step")
                }
            }
        }

        override fun onReceiveWeightHistoryData(device: ICDevice, data: ICWeightHistoryData) {
            Log.d(TAG, "Received weight history data from ${device.macAddr}")

            val historyMap = Arguments.createMap().apply {
                putDouble("weight", data.weight_kg.toDouble())
                // Remove measureTime as it doesn't exist in this SDK version
                putLong("timestamp", System.currentTimeMillis())
            }

            val params = Arguments.createMap().apply {
                putString("mac", device.macAddr)
                putMap("data", historyMap)
            }
            emitToJS("onReceiveWeightHistoryData", params)
        }
        
        override fun onReceiveWeightUnitChanged(device: ICDevice, unit: ICConstant.ICWeightUnit) {}

        // Kitchen scale callbacks
        override fun onReceiveKitchenScaleData(device: ICDevice, data: ICKitchenScaleData) {}
        override fun onReceiveKitchenScaleUnitChanged(device: ICDevice, unit: ICConstant.ICKitchenScaleUnit) {}

        // Ruler callbacks
        override fun onReceiveRulerData(device: ICDevice, data: ICRulerData) {}
        override fun onReceiveRulerHistoryData(device: ICDevice, data: ICRulerData) {}
        override fun onReceiveRulerUnitChanged(device: ICDevice, unit: ICConstant.ICRulerUnit) {}
        override fun onReceiveRulerMeasureModeChanged(device: ICDevice, mode: ICConstant.ICRulerMeasureMode) {}

        // Skipping rope callbacks
        override fun onReceiveSkipData(device: ICDevice, data: ICSkipData) {}
        override fun onReceiveHistorySkipData(device: ICDevice, data: ICSkipData) {}
        override fun onNodeConnectionChanged(device: ICDevice, nodeId: Int, state: ICConstant.ICDeviceConnectState) {}

        // Callbacks not in docs
        override fun onReceiveDebugData(device: ICDevice, type: Int, data: Any?) {}
        override fun onReceiveDeviceLightSetting(device: ICDevice, data: Any?) {}
        override fun onReceiveScanWifiInfo_W(device: ICDevice, ssid: String, method: Int, rssi: Int) {}
        override fun onReceiveCurrentWifiInfo_W(device: ICDevice, status: Int, ip: String, ssid: String, rssi: Int) {}
        override fun onReceiveBindState_W(device: ICDevice, status: Int) {}
        override fun onReceiveConfigWifiResult(device: ICDevice, state: ICConstant.ICConfigWifiState) {}
    }

    private val scanDelegate = object : ICScanDeviceDelegate {
        override fun onScanResult(deviceInfo: ICScanDeviceInfo) {
            Log.d(TAG, "Found device: ${deviceInfo.macAddr}")

            if (!scannedDevices.containsKey(deviceInfo.macAddr)) {
                val device = ICDevice()
                device.setMacAddr(deviceInfo.macAddr)
                scannedDevices[deviceInfo.macAddr] = device

                Log.d(TAG, "Added new device to scanned list: ${device.macAddr}")
                emitToJS("onDeviceFound", Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                    putInt("rssi", deviceInfo.rssi)
                })
            }
        }
    }

    // -------------------- HELPERS --------------------

    private fun handleWeightData(device: ICDevice, data: ICWeightData) {
        Log.d(TAG, "Handling weight data from ${device.macAddr}: ${data.weight_kg}kg")

        val weightMap = Arguments.createMap().apply {
            putDouble("weight", data.weight_kg.toDouble())
            putLong("timestamp", System.currentTimeMillis())
            putBoolean("isStabilized", data.isStabilized)
            putInt("heartRate", data.hr)
        }

        val deviceMap = Arguments.createMap().apply {
            putString("mac", device.macAddr)
        }

        val params = Arguments.createMap().apply {
            putMap("device", deviceMap)
            putMap("data", weightMap)
        }
        emitToJS("onReceiveWeightData", params)
    }

    private fun handleWeightCenterData(device: ICDevice, data: ICWeightCenterData) {
        Log.d(TAG, "Handling weight center data from ${device.macAddr}")

        val centerMap = Arguments.createMap().apply {
            putDouble("weight", data.precision_kg.toDouble())
            putLong("timestamp", System.currentTimeMillis())
        }

        val params = Arguments.createMap().apply {
            putString("mac", device.macAddr)
            putMap("data", centerMap)
        }
        emitToJS("onReceiveWeightCenterData", params)
    }

    private fun emitToJS(event: String, params: Any?) {
        if (!hasListeners()) return
        reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
            .emit(event, params)
    }

    private val bluetoothReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val state = intent?.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
            val stateName = when (state) {
                BluetoothAdapter.STATE_ON -> "ICBleStatePoweredOn"
                BluetoothAdapter.STATE_OFF -> "ICBleStatePoweredOff"
                else -> "ICBleStateUnknown"
            }
            emitToJS("onBleState", Arguments.createMap().apply { putString("state", stateName) })
        }
    }

    fun setUserList(userInfoMap: ReadableMap) {
        val userList = mutableListOf<ICUserInfo>()
        Log.d(TAG, "Set user list")
        val userInfo = ICUserInfo()
        Log.d(TAG, "Original user info: $userInfo")
        userInfo.nickName = userInfoMap.getString("name") ?: ""
        userInfo.age = userInfoMap.getInt("age")
        userInfo.height = userInfoMap.getInt("height")
        val genderStr = userInfoMap.getString("gender") ?: "MALE"
        userInfo.sex = when (genderStr.uppercase()) {
            "FEMALE" -> ICConstant.ICSexType.ICSexTypeFemal
            "MALE" -> ICConstant.ICSexType.ICSexTypeMale
            else -> ICConstant.ICSexType.ICSexTypeMale
        }
        userInfo.peopleType = ICConstant.ICPeopleType.ICPeopleTypeNormal
        userList.add(userInfo)
        Log.d(TAG, "Setting user list: $userList")
        deviceManager.setUserList(userList)
    }

    fun restoreConnectedDevices() {
        val prefs = reactContext.getSharedPreferences("ConnectedDevices", Context.MODE_PRIVATE)
        val macs = prefs.all.keys
        macs.forEach { mac ->
            val device = ICDevice()
            device.setMacAddr(mac)
            deviceManager.addDevice(device) { d, code ->
                if (code == ICConstant.ICAddDeviceCallBackCode.ICAddDeviceCallBackCodeSuccess) {
                    connectedDevices[mac] = d
                    Log.d(TAG, "Restored connected device: $mac")
                }
            }
        }
    }

    // -------------------- ACCESSORS --------------------

    fun getConnectedDevice(mac: String): ICDevice? {
        return connectedDevices[mac];
    }

    fun getAllConnectedDevices(): Collection<ICDevice> {
        return connectedDevices.values
    }

    fun cleanup() {
        if (receiverRegistered) {
            reactContext.unregisterReceiver(bluetoothReceiver)
            receiverRegistered = false
        }
    }
}
