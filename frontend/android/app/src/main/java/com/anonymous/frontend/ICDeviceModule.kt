package com.anonymous.frontend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.ActivityCompat
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

class ICDeviceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val TAG = "ICDeviceModule"
        private const val CODE_REQUEST_PERMISSION = 1001
    }

    private val scannedDevices = mutableMapOf<String, ICDevice>()
    private val connectedDevices = mutableMapOf<String, ICDevice>()
    private var listenerCount = 0
    private var isScanning = false
    private var isSDKInitialized = false

    private var deviceManagerDelegate: ICDeviceManagerDelegate? = null
    private var scanDelegate: ICScanDeviceDelegate? = null

    init {
        setupDelegates()
    }

    override fun getName(): String = "ICDeviceModule"

    @ReactMethod
    fun addListener(eventName: String) {
        listenerCount++
        Log.d(TAG, "Added listener for $eventName, total: $listenerCount")
    }

    @ReactMethod
    fun removeListeners(count: Double) {
        listenerCount -= count.toInt()
        if (listenerCount < 0) listenerCount = 0
        Log.d(TAG, "Removed ${count.toInt()} listeners, remaining: $listenerCount")
    }

    private fun emitToJS(event: String, params: Any?) {
        if (listenerCount > 0) {
            reactContext
                .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java)
                ?.emit(event, params)
        }
    }

    private fun checkBLEPermission(): Boolean {
        val activity = reactContext.currentActivity ?: return false

        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ActivityCompat.checkSelfPermission(
                activity,
                Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        activity,
                        Manifest.permission.BLUETOOTH_SCAN
                    ) == PackageManager.PERMISSION_GRANTED
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.checkSelfPermission(
                activity,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED &&
                    ActivityCompat.checkSelfPermission(
                        activity,
                        Manifest.permission.ACCESS_COARSE_LOCATION
                    ) == PackageManager.PERMISSION_GRANTED
        } else {
            true
        }
    }

    private fun requestBLEPermission() {
        val activity = reactContext.currentActivity ?: return

        val permissions = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            arrayOf(
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN
            )
        } else {
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            )
        }

        ActivityCompat.requestPermissions(
            activity,
            permissions,
            CODE_REQUEST_PERMISSION
        )
    }

    private fun setupDelegates() {
        deviceManagerDelegate = object : ICDeviceManagerDelegate {
            override fun onInitFinish(success: Boolean) {
                Log.d(TAG, "SDK init finished: $success")
                isSDKInitialized = success
                emitToJS("onSDKInit", Arguments.createMap().apply {
                    putBoolean("success", success)
                })
            }

            override fun onBleState(state: ICConstant.ICBleState) {
                Log.d(TAG, "BLE state: $state")
                emitToJS("onBleState", Arguments.createMap().apply {
                    putString("state", state.name)
                    putBoolean("enabled", state == ICConstant.ICBleState.ICBleStatePoweredOn)
                })
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

            // Remove the old onReceiveWeightData method as it's now handled by onReceiveMeasureStepData

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

            override fun onReceiveRSSI(device: ICDevice, rssi: Int) {
                val params = Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                    putInt("rssi", rssi)
                }
                emitToJS("onReceiveRSSI", params)
            }

            // Empty implementations for other required methods
            override fun onReceiveWeightData(device: ICDevice, data: ICWeightData) {
                // This method is now handled by onReceiveMeasureStepData
                // Keep this override for interface compliance but delegate to the new method
                handleWeightData(device, data)
            }
            override fun onReceiveUserInfo(device: ICDevice, userInfo: ICUserInfo) {}
            override fun onReceiveUserInfoList(device: ICDevice, list: List<ICUserInfo>) {}
            override fun onReceiveHR(device: ICDevice, hr: Int) {}
            override fun onReceiveWeightCenterData(device: ICDevice, data: ICWeightCenterData) {
                // This method is now handled by onReceiveMeasureStepData
                handleWeightCenterData(device, data)
            }
            override fun onReceiveCoordData(device: ICDevice, data: ICCoordData) {}
            override fun onReceiveWeightUnitChanged(device: ICDevice, unit: ICConstant.ICWeightUnit) {}
            override fun onReceiveKitchenScaleData(device: ICDevice, data: ICKitchenScaleData) {}
            override fun onReceiveKitchenScaleUnitChanged(device: ICDevice, unit: ICConstant.ICKitchenScaleUnit) {}
            override fun onReceiveRulerData(device: ICDevice, data: ICRulerData) {}
            override fun onReceiveRulerHistoryData(device: ICDevice, data: ICRulerData) {}
            override fun onReceiveRulerUnitChanged(device: ICDevice, unit: ICConstant.ICRulerUnit) {}
            override fun onReceiveRulerMeasureModeChanged(device: ICDevice, mode: ICConstant.ICRulerMeasureMode) {}
            override fun onReceiveSkipData(device: ICDevice, data: ICSkipData) {}
            override fun onReceiveHistorySkipData(device: ICDevice, data: ICSkipData) {}
            override fun onNodeConnectionChanged(device: ICDevice, nodeId: Int, state: ICConstant.ICDeviceConnectState) {}
            override fun onReceiveDebugData(device: ICDevice, type: Int, data: Any?) {}
            override fun onReceiveDeviceLightSetting(device: ICDevice, data: Any?) {}
            override fun onReceiveScanWifiInfo_W(device: ICDevice, ssid: String, method: Int, rssi: Int) {}
            override fun onReceiveCurrentWifiInfo_W(device: ICDevice, status: Int, ip: String, ssid: String, rssi: Int) {}
            override fun onReceiveBindState_W(device: ICDevice, status: Int) {}
            override fun onReceiveConfigWifiResult(device: ICDevice, state: ICConstant.ICConfigWifiState) {}
        }

        scanDelegate = object : ICScanDeviceDelegate {
            override fun onScanResult(deviceInfo: ICScanDeviceInfo) {
                Log.d(TAG, "Found device: ${deviceInfo.macAddr}")

                if (!scannedDevices.containsKey(deviceInfo.macAddr)) {
                    val device = ICDevice()
                    device.setMacAddr(deviceInfo.macAddr)
                    scannedDevices[deviceInfo.macAddr] = device

                    Log.d(TAG, "Added new device to scanned list: ${device.macAddr}")

                    val params = Arguments.createMap().apply {
                        putString("mac", device.macAddr)
                        putInt("rssi", deviceInfo.rssi)
                    }
                    emitToJS("onDeviceFound", params)
                }
            }
        }
    }

    @ReactMethod
    fun initializeSDK(promise: Promise) {
        val activity = reactContext.currentActivity
        if (activity == null) {
            Log.w(TAG, "Activity is null, cannot init SDK yet")
            promise.reject("NO_ACTIVITY", "Activity is null")
            return
        }

        if (!checkBLEPermission()) {
            requestBLEPermission()
            promise.reject("NO_PERMISSION", "BLE permissions not granted")
            return
        }

        try {
            val config = ICDeviceManagerConfig()
            config.context = reactContext.applicationContext
            deviceManagerDelegate?.let { delegate ->
                ICDeviceManager.shared().setDelegate(delegate)
            }
            ICDeviceManager.shared().initMgrWithConfig(config)
            Log.d(TAG, "SDK initialization started")
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize SDK: ${e.message}")
            promise.reject("INIT_ERROR", e.message)
        }
    }

    @ReactMethod
    fun startScan(promise: Promise) {
        if (!isSDKInitialized) {
            promise.reject("SDK_NOT_INITIALIZED", "SDK not initialized")
            return
        }

        if (!checkBLEPermission()) {
            requestBLEPermission()
            promise.reject("NO_PERMISSION", "BLE permissions not granted")
            return
        }

        if (isScanning) {
            promise.resolve(false)
            return
        }

        try {
            scannedDevices.clear()

            scanDelegate?.let { delegate ->
                ICDeviceManager.shared().scanDevice(delegate)
                isScanning = true
                Log.d(TAG, "Scan started")
                promise.resolve(true)
            } ?: run {
                promise.reject("NO_DELEGATE", "Scan delegate not set")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start scan: ${e.message}")
            promise.reject("SCAN_ERROR", e.message)
        }
    }

    @ReactMethod
    fun stopScan(promise: Promise) {
        try {
            ICDeviceManager.shared().stopScan()
            isScanning = false
            Log.d(TAG, "Scan stopped")
            promise.resolve(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop scan: ${e.message}")
            promise.reject("STOP_SCAN_ERROR", e.message)
        }
    }

    @ReactMethod
    fun connectDevice(macAddress: String, promise: Promise) {
        try {
            val device = scannedDevices[macAddress]
            if (device == null) {
                promise.reject("DEVICE_NOT_FOUND", "Device with MAC $macAddress not found in scanned devices")
                return
            }

            if (connectedDevices.containsKey(macAddress)) {
                promise.resolve(true)
                return
            }

            Log.d(TAG, "Attempting to add/connect device: $macAddress")
            ICDeviceManager.shared().addDevice(device, object : ICConstant.ICAddDeviceCallBack {
                override fun onCallBack(device: ICDevice, code: ICConstant.ICAddDeviceCallBackCode) {
                    Log.d(TAG, "Device add callback: ${device.macAddr} -> $code")
                    when (code) {
                        ICConstant.ICAddDeviceCallBackCode.ICAddDeviceCallBackCodeSuccess -> {
                            promise.resolve(true)
                        }
                        else -> {
                            // Handle all other cases as potential success or existing device
                            Log.d(TAG, "Device add result: $code")
                            promise.resolve(true)
                        }
                    }
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error connecting to device: ${e.message}")
            promise.reject("CONNECTION_ERROR", e.message)
        }
    }

    @ReactMethod
    fun disconnectDevice(macAddress: String, promise: Promise) {
        try {
            val device = connectedDevices[macAddress] ?: scannedDevices[macAddress]
            if (device == null) {
                promise.reject("DEVICE_NOT_FOUND", "Device with MAC $macAddress not found")
                return
            }

            Log.d(TAG, "Removing device: $macAddress")
            ICDeviceManager.shared().removeDevice(device, object : ICConstant.ICRemoveDeviceCallBack {
                override fun onCallBack(device: ICDevice, code: ICConstant.ICRemoveDeviceCallBackCode) {
                    Log.d(TAG, "Device remove callback: ${device.macAddr} -> $code")
                    when (code) {
                        ICConstant.ICRemoveDeviceCallBackCode.ICRemoveDeviceCallBackCodeSuccess -> {
                            connectedDevices.remove(macAddress)
                            promise.resolve(true)
                        }
                        else -> {
                            // Handle all other cases - log and resolve as success
                            Log.d(TAG, "Device remove result: $code")
                            connectedDevices.remove(macAddress)
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

    @ReactMethod
    fun getScannedDevices(promise: Promise) {
        try {
            val devicesArray = Arguments.createArray()
            scannedDevices.values.forEach { device ->
                devicesArray.pushMap(Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                    putBoolean("isConnected", connectedDevices.containsKey(device.macAddr))
                })
            }
            promise.resolve(devicesArray)
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }

    @ReactMethod
    fun getConnectedDevices(promise: Promise) {
        try {
            val devicesArray = Arguments.createArray()
            connectedDevices.values.forEach { device ->
                devicesArray.pushMap(Arguments.createMap().apply {
                    putString("mac", device.macAddr)
                    putBoolean("isConnected", true)
                })
            }
            promise.resolve(devicesArray)
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }

    @ReactMethod
    fun isDeviceConnected(macAddress: String, promise: Promise) {
        promise.resolve(connectedDevices.containsKey(macAddress))
    }

    @ReactMethod
    fun isScanning(promise: Promise) {
        promise.resolve(isScanning)
    }

    @ReactMethod
    fun isSDKInitialized(promise: Promise) {
        promise.resolve(isSDKInitialized)
    }

    @ReactMethod
    fun clearScannedDevices(promise: Promise) {
        scannedDevices.clear()
        promise.resolve(true)
    }

    @ReactMethod
    fun getBodyFatAlgorithmsManager(promise: Promise) {
        try {
            promise.resolve(null)
        } catch (e: Exception) {
            promise.reject("ERROR", e.message)
        }
    }
}