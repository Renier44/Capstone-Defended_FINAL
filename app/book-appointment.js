import { Ionicons } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { useState, useEffect } from 'react';
import {
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  ActivityIndicator,
  Alert,
} from 'react-native';

export default function BookAppointment() {
  const router = useRouter();
  const { reason: aiReason, aiScreening } = useLocalSearchParams();

  // Initial state for patient details
  const [bookingFor, setBookingFor] = useState(''); // Initial value will be set by fetched data
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [dob, setDob] = useState('');
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [gender, setGender] = useState('');
  const [age, setAge] = useState('');
  const [reason, setReason] = useState('');

  // State for fetched data and loading status
  const [reasonsList, setReasonsList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [genderChoices, setGenderChoices] = useState([]);
  const [loadingGenders, setLoadingGenders] = useState(true);
  
  // NEW state variables for Booking For
  const [bookingForChoices, setBookingForChoices] = useState([]);
  const [loadingBookingFor, setLoadingBookingFor] = useState(true);

  // Your ngrok URL
  const API_BASE_URL ='https://2b7bf55b1e09.ngrok-free.app';

  useEffect(() => {
    // Function to fetch reasons
    const fetchReasons = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/reasons/`);
        if (!response.ok) {
          throw new Error('Failed to fetch appointment reasons.');
        }
        const data = await response.json();
        setReasonsList(data);
      } catch (error) {
        console.error('Error fetching reasons:', error);
        Alert.alert('Error', 'Failed to load reasons. Please check your connection.');
      } finally {
        setLoading(false);
      }
    };

    if (aiScreening && aiReason) {
      setReason(aiReason);
      setLoading(false);
    } else {
      fetchReasons();
    }
  }, [aiScreening, aiReason]);

  useEffect(() => {
    const fetchGenderChoices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/genders/`);
        if (!response.ok) {
          throw new Error('Failed to fetch gender choices.');
        }
        const data = await response.json();
        setGenderChoices(data);
        if (data.length > 0) {
          setGender(data[0]);
        }
      } catch (error) {
        console.error('Error fetching genders:', error);
        Alert.alert('Error', 'Failed to load gender options.');
      } finally {
        setLoadingGenders(false);
      }
    };

    fetchGenderChoices();
  }, []);

  // NEW useEffect to fetch booking for options
  useEffect(() => {
    const fetchBookingForChoices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/appointment/booking-for/`);
        if (!response.ok) {
          throw new Error('Failed to fetch booking for choices.');
        }
        const data = await response.json();
        setBookingForChoices(data);
        if (data.length > 0) {
          setBookingFor(data[0]);
        }
      } catch (error) {
        console.error('Error fetching booking for:', error);
        Alert.alert('Error', 'Failed to load booking for options.');
      } finally {
        setLoadingBookingFor(false);
      }
    };

    fetchBookingForChoices();
  }, []);

  const calculateAge = (birthDate) => {
    const today = new Date();
    let years = today.getFullYear() - birthDate.getFullYear();
    const m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
      years--;
    }
    return years;
  };

  const onChangeDate = (event, selectedDate) => {
    const currentDate = selectedDate || new Date(dob || new Date());
    if (Platform.OS === 'android') setShowDatePicker(false);

    const formattedDate = `${currentDate.getFullYear()}-${String(
      currentDate.getMonth() + 1
    ).padStart(2, '0')}-${String(currentDate.getDate()).padStart(2, '0')}`;

    setDob(formattedDate);
    setAge(calculateAge(currentDate).toString());
  };

  const handleNext = () => {
    if (!firstName || !lastName || !dob || !age || !reason || !gender) {
      Alert.alert('Missing Details', 'Please fill all details.');
      return;
    }

    router.push({
      pathname: '/schedule',
      params: {
        firstName,
        lastName,
        dob,
        gender,
        age: parseInt(age),
        reason,
        bookingFor,
      },
    });
  };

  const formatText = (text) => {
    // A helper function to format the text for display
    return text.charAt(0).toUpperCase() + text.slice(1).replace('-', ' ');
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={24} color="#000" />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: '#000' }]}>Patient Details</Text>
      </View>
      <View style={styles.content}>
        {/* Booking For - Updated to use fetched data */}
        <Text style={styles.label}>Booking For</Text>
        {loadingBookingFor ? (
          <ActivityIndicator size="small" color="#1E88E5" />
        ) : (
          <View style={styles.toggleWrapper}>
            {bookingForChoices.map((option) => (
              <TouchableOpacity
                key={option}
                style={[styles.toggleOption, bookingFor === option && styles.toggleOptionActive]}
                onPress={() => setBookingFor(option)}
              >
                <Text style={[styles.toggleText, bookingFor === option && styles.toggleTextActive]}>
                  {formatText(option)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <View style={styles.inputRow}>
          <Text style={styles.label}>Name:</Text>
          <TextInput style={styles.inputField} placeholder="First Name" value={firstName} onChangeText={setFirstName} />
        </View>
        <View style={styles.inputRow}>
          <Text style={styles.label}>Last Name:</Text>
          <TextInput style={styles.inputField} placeholder="Last Name" value={lastName} onChangeText={setLastName} />
        </View>

        <View style={styles.inputRow}>
          <Text style={styles.label}>Date of Birth:</Text>
          <TouchableOpacity style={styles.inputField} onPress={() => setShowDatePicker(true)}>
            <TextInput
              style={{ padding: 0, margin: 0 }}
              placeholder="MM-DD-YYYY"
              value={dob}
              editable={false}
            />
          </TouchableOpacity>
        </View>

        {showDatePicker && (
          <DateTimePicker
            value={dob ? new Date(dob) : new Date()}
            mode="date"
            display={Platform.OS === 'ios' ? 'spinner' : 'default'}
            onChange={onChangeDate}
            maximumDate={new Date()}
          />
        )}

        {/* Gender Selection - Updated to use fetched data */}
        <Text style={styles.label}>Gender:</Text>
        {loadingGenders ? (
          <ActivityIndicator size="small" color="#1E88E5" />
        ) : (
          <View style={styles.genderWrapper}>
            {genderChoices.map((option) => (
              <TouchableOpacity
                key={option}
                style={[styles.genderOption, gender === option && styles.genderOptionActive]}
                onPress={() => setGender(option)}
              >
                <Text style={[styles.genderText, gender === option && styles.genderTextActive]}>
                  {option}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <View style={styles.inputRow}>
          <Text style={styles.label}>Age:</Text>
          <TextInput style={styles.inputFieldAge} placeholder="Age" value={age} editable={false} />
        </View>

        <Text style={styles.label}>Reason for Appointment</Text>
        {loading ? (
          <ActivityIndicator size="large" color="#1E88E5" />
        ) : aiScreening ? (
          <TextInput
            style={[styles.reasonInput, { backgroundColor: '#C8EAF7' }]}
            value={reason}
            editable={false}
          />
        ) : (
          <View style={styles.reasonContainer}>
            {reasonsList.map((item, idx) => (
              <TouchableOpacity
                key={idx}
                style={[styles.reasonItem, reason === item && styles.reasonItemSelected]}
                onPress={() => setReason(item)}
              >
                <Text style={[styles.reasonText, reason === item && styles.reasonTextSelected]}>{item}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        <Text style={styles.noteText}>
          Note: For Users 18 And Under, Book With Dr. Mikaela Alvarez. For Users 19 And Older, Book With Dr. Cherry Alvarez.
        </Text>

        <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
          <Text style={styles.nextButtonText}>Next</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#77CDE0',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'transparent',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginLeft: 15,
    fontFamily: 'League Spartan',
    color: '#2260FF',
    textAlign: 'center',
    width: '100%',
  },
  content: {
    paddingHorizontal: 20,
    paddingBottom: 40,
  },
  label: {
    fontSize: 15,
    fontFamily: 'League Spartan',
    marginBottom: 8,
    color: '#000',
  },
  toggleWrapper: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 30,
    marginBottom: 20,
    padding: 4,
  },
  toggleOption: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 25,
  },
  toggleOptionActive: {
    backgroundColor: '#FFD54F',
  },
  toggleText: {
    fontSize: 14,
    fontFamily: 'League Spartan',
    color: '#000',
  },
  toggleTextActive: {
    color: '#000',
    fontWeight: 'bold',
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    marginBottom: 15,
  },
  inputField: {
    borderWidth: 1,
    borderColor: '#bdc3c7',
    padding: 6,
    borderRadius: 30,
    fontSize: 15,
    fontFamily: 'League Spartan',
    backgroundColor: '#C8EAF7',
    width: 150,
    marginLeft: 10,
    height: 40,
  },
  inputFieldAge: {
    borderWidth: 1,
    borderColor: '#bdc3c7',
    padding: 6,
    borderRadius: 30,
    fontSize: 15,
    fontFamily: 'League Spartan',
    backgroundColor: '#C8EAF7',
    width: 80,
    marginLeft: 10,
    height: 40,
    textAlign: 'center',
  },
  genderWrapper: {
    flexDirection: 'row',
    marginBottom: 20,
    justifyContent: 'space-around',
  },
  genderOption: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#bdc3c7',
    borderRadius: 30,
    marginHorizontal: 5,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
  },
  genderOptionActive: {
    backgroundColor: '#FFD54F',
    borderColor: '#FFD54F',
  },
  genderText: {
    fontFamily: 'League Spartan',
    fontSize: 15,
    color: '#000',
  },
  genderTextActive: {
    color: '#000',
    fontWeight: 'bold',
  },
  reasonContainer: {
    backgroundColor: '#C8EAF7',
    padding: 10,
    borderRadius: 30,
    marginBottom: 15,
  },
  reasonInput: {
    borderWidth: 1,
    borderColor: '#bdc3c7',
    padding: 12,
    borderRadius: 30,
    fontSize: 15,
    fontFamily: 'League Spartan',
    marginBottom: 15,
    backgroundColor: '#C8EAF7',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  reasonItem: {
    padding: 10,
    borderRadius: 30,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#bdc3c7',
    alignItems: 'center',
    width: '100%',
    backgroundColor: '#A5E3F0',
  },
  reasonItemSelected: {
    backgroundColor: '#FFD54F',
    borderColor: '#FFD54F',
  },
  reasonText: {
    fontFamily: 'League Spartan',
    fontSize: 15,
    color: '#000',
  },
  reasonTextSelected: {
    color: '#000',
    fontWeight: 'bold',
  },
  noteText: {
    fontSize: 12,
    color: '#555',
    textAlign: 'center',
    marginVertical: 20,
    fontFamily: 'League Spartan',
  },
  nextButton: {
    backgroundColor: '#FFD54F',
    padding: 15,
    borderRadius: 30,
    alignItems: 'center',
    marginTop: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  nextButtonText: {
    color: '#000',
    fontSize: 18,
    fontWeight: 'bold',
    fontFamily: 'League Spartan',
  },
});
