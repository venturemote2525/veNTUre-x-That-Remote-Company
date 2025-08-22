import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';
import { useRouter } from 'expo-router';
import { Pressable } from 'react-native';

export default function HomeScreen() {
  const router = useRouter();
  return (
    <ThemedSafeAreaView className="p-4">
      <View className="flex-row justify-between">
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
            onPress={() => router.push('/(user)/MyDevices')}
            itemTextClassName="text-primary-500"
          />
          <DropdownItem label={'Manual Input'} />
        </CustomDropdown>
      </View>
    </ThemedSafeAreaView>
  );
}
