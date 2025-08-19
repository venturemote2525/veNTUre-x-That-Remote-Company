import CustomDropdown, { DropdownItem } from '@/components/CustomDropdown';
import { ThemedSafeAreaView, Text, View } from '@/components/Themed';
import { AddIcon, Icon } from '@/components/ui/icon';

export default function HomeScreen() {
  return (
    <ThemedSafeAreaView className="p-4">
      <View className="flex-row justify-between">
        <Text className="text-head2 font-heading text-secondary-500">
          HealthSync
        </Text>
        <CustomDropdown
          toggle={
            <Icon as={AddIcon} size={'xl'} className="text-secondary-500" />
          }>
          <DropdownItem label={'Add Device'} />
          <DropdownItem label={'Manual Input'} />
        </CustomDropdown>
      </View>
    </ThemedSafeAreaView>
  );
}
