import { Modal, Pressable } from 'react-native';
import { View, Text } from '@/components/Themed';

type CustomAlertProps = {
  visible: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel?: () => void;
  confirmText?: string;
  cancelText?: string;
};

export function CustomAlert({
  visible,
  title,
  message,
  onConfirm,
  onCancel,
  confirmText = 'OK',
  cancelText = 'Cancel',
}: CustomAlertProps) {
  return (
    <Modal transparent visible={visible} animationType="fade">
      <View
        className="flex-1 items-center justify-center"
        style={{ backgroundColor: 'rgba(0,0,0,0.3)' }}>
        <View className="w-[80%] gap-4 rounded-2xl bg-background-0 p-6">
          <View className="gap-1">
            <Text className="font-heading text-body1 text-secondary-500">
              {title}
            </Text>
            <Text className="text-body2 text-primary-500">{message}</Text>
          </View>
          <View className="flex-row gap-4">
            {onCancel && (
              <Pressable
                className="flex-1 items-center justify-center rounded-lg bg-[#00AFB9] p-2"
                onPress={onCancel}>
                <Text className="font-bodyBold text-background-0">
                  {cancelText}
                </Text>
              </Pressable>
            )}
            <Pressable
              className="flex-1 items-center justify-center rounded-lg bg-error-500 p-2"
              onPress={onConfirm}>
              <Text className="font-bodyBold text-background-0">
                {confirmText}
              </Text>
            </Pressable>
          </View>
        </View>
      </View>
    </Modal>
  );
}
