import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { Pressable, useColorScheme } from 'react-native';
import { faUtensils, faChild } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import { Colors } from '@/constants/Colors';
import { useEffect, useState } from 'react';
import { useICDevice } from '@/context/ICDeviceContext';
import { AlertState } from '@/types/database-types';
import { CustomAlert } from '@/components/CustomAlert';

export default function HomeScreen() {
  const router = useRouter();
  const rawScheme = useColorScheme();
  const scheme: 'light' | 'dark' = rawScheme === 'dark' ? 'dark' : 'light';

  const { bleEnabled } = useICDevice();
  const [alert, setAlert] = useState<AlertState>({
    visible: false,
    title: '',
    message: '',
  });

  useEffect(() => {
    if (!bleEnabled) {
      setAlert({
        visible: true,
        title: 'Bluetooth not enabled',
        message: 'Please enable Bluetooth to use weight scales.',
      });
    }
  }, []);

  return (
    <ThemedSafeAreaView>
      {/* Header */}
      <View className="flex-row justify-between p-4">
        <Text className="font-heading text-head2 text-secondary-500">
          HealthSync
        </Text>
        <CustomDropdown
          toggle={
            <Pressable>
              <Icon as={AddIcon} size={'xl'} className="text-secondary-500" />
            </Pressable>
          }
          menuClassName="min-w-40 rounded-2xl bg-background-0 p-3"
          separator={true}>
          <DropdownItem
            label={'Add Device'}
            onPress={() => router.push('/(device)/MyDevices')}
            itemTextClassName="text-primary-500"
          />
          <DropdownItem label={'Manual Input'} />
        </CustomDropdown>
      </View>
      {/* Main */}
      <View className="flex-1 gap-4 px-4">
        {/* Food */}
        <Pressable
          onPress={() => router.push('/(tabs)/food')}
          className="card-white flex-row justify-between">
          {/* Title */}
          <View className="flex-row items-center gap-4">
            <View className="h-10 w-10 items-center justify-center rounded-full bg-secondary-500">
              <FontAwesomeIcon
                icon={faUtensils}
                size={18}
                color={Colors[scheme].background}
              />
            </View>
            <Text className="font-bodyBold text-body1 text-primary-500">
              Food
            </Text>
          </View>
          <View className="items-center">
            <Text className="font-heading text-body1 text-primary-500">10</Text>
            <Text className="font-bodyBold text-body2 text-primary-500">
              kcal
            </Text>
          </View>
        </Pressable>
        {/* Body */}
        <Pressable
          onPress={() => router.push('/(tabs)/body')}
          className="card-white gap-4">
          {/* Title */}
          <View className="flex-row items-center gap-4">
            <View className="h-10 w-10 items-center justify-center rounded-full bg-secondary-500">
              <FontAwesomeIcon
                icon={faChild}
                size={18}
                color={Colors[scheme].background}
              />
            </View>
            <Text className="font-bodyBold text-body1 text-primary-500">
              Body Composition
            </Text>
          </View>
          {/* Details */}
          <View>
            <Text>Body Fat:</Text>
            <Text>Weight:</Text>
          </View>
        </Pressable>
      </View>

      <CustomAlert
        visible={alert.visible}
        title={alert.title}
        message={alert.message}
        onConfirm={() => {
          setAlert({ ...alert, visible: false });
        }}
      />
    </ThemedSafeAreaView>
  );
}
