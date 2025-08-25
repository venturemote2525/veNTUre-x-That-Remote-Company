# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /usr/local/Cellar/android-sdk/24.3.3/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# react-native-reanimated
-keep class com.swmansion.reanimated.** { *; }
-keep class com.facebook.react.turbomodule.** { *; }

# Add any project specific keep options here:

-keep class cn.icomon.icdevicemanager.ICDeviceManager { *; }
-keep class cn.icomon.icdevicemanager.ICBluetoothSystem { *; }
-keep public interface cn.icomon.icdevicemanager.ICBluetoothSystem$ICBluetoothDelegate { *; }
-keep public class cn.icomon.icdevicemanager.ICBluetoothSystem$ICOPBleCharacteristic { *; }
-keep public enum cn.icomon.icdevicemanager.ICBluetoothSystem$ICOPBleWriteDataType { *; }
-keep class cn.icomon.icdevicemanager.manager.setting.ICSettingManagerImpl { *; }
-keep class cn.icomon.icdevicemanager.manager.algorithms.ICBodyFatAlgorithmsImpl { *; }
-keep class cn.icomon.icdevicemanager.ICDeviceManagerDelegate { *; }
-keep class cn.icomon.icdevicemanager.model.** { *; }
-keep class cn.icomon.icdevicemanager.ICDeviceManagerSettingManager { *; }
-keep public interface cn.icomon.icdevicemanager.ICDeviceManagerSettingManager$ICSettingCallback { *; }
-keep class com.icomon.icbodyfatalgorithms.** { *; }
-keep class cn.icomon.icbleprotocol.** { *; }
-keep class cn.icomon.icdevicemanager.ICBodyFatAlgorithmsManager { *; }
-keep class cn.icomon.icdevicemanager.ICBluetoothSystem.** { *; }
-keep class cn.icomon.icdevicemanager.callback.** { *; }