import uuid from 'react-native-uuid';
import AsyncStorage from '@react-native-async-storage/async-storage';

export const calculateAge = (dob: string | Date): number => {
  const birthDate = new Date(dob);
  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  if (
    monthDiff < 0 ||
    (monthDiff === 0 && today.getDate() < birthDate.getDate())
  ) {
    age--;
  }
  return age;
};

/**
 * Get the phone ID from AsyncStorage. If it doesn't exist, generate a new one and save it.
 * To identify device user has connected to scale
 * @returns {Promise<string>} The phone ID.
 */
export async function getPhoneId(): Promise<string> {
  let id = await AsyncStorage.getItem('phoneId');
  if (!id) {
    id = uuid.v4();
    await AsyncStorage.setItem('phoneId', id);
  }
  return id;
}
