import { View, Text } from '@/components/Themed';
import { ChevronLeftIcon, Icon } from '@/components/ui/icon';
import { useNavigation } from 'expo-router';
import { Pressable } from 'react-native';

type HeaderProps = {
  title?: string;
  onBackPress?: () => void;
};

export default function Header({ title, onBackPress }: HeaderProps) {
  const navigation = useNavigation();
  return (
    <View className="flex-row items-center gap-2 px-2 py-3">
      <Pressable
        onPress={() => {
          if (onBackPress) onBackPress();
          else navigation.goBack();
        }}>
        <Icon as={ChevronLeftIcon} size={'xl'} className="text-secondary-500" />
      </Pressable>
      <Text className="font-bodyBold text-2xl text-secondary-500">{title}</Text>
    </View>
  );
}
