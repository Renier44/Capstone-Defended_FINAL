import { Ionicons } from '@expo/vector-icons';
import { useLocalSearchParams, useRouter } from 'expo-router';
import {
  Image,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

export default function AppointmentSuccess() {
  const router = useRouter();
  const params = useLocalSearchParams();

  // Retrieve all the parameters passed from the previous screen
  const {
    firstName,
    lastName,
    age,
    gender,
    reason,
    bookingFor,
    assignedDoctorName,
    appointmentDate,
    selectedTime,
  } = params;

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>Appointment Booked</Text>
        <Text style={styles.subHeaderText}>Your appointment has been successfully booked.</Text>
      </View>

      <View style={styles.card}>
        <View style={styles.iconContainer}>
          <Ionicons name="checkmark-circle" size={80} color="#4CAF50" />
        </View>

        <Text style={styles.cardTitle}>Appointment Details</Text>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Doctor Assigned</Text>
          <Text style={styles.value}>{assignedDoctorName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Appointment Date</Text>
          <Text style={styles.value}>{appointmentDate}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Time</Text>
          <Text style={styles.value}>{selectedTime}</Text>
        </View>

        <View style={styles.divider} />

        <View style={styles.detailRow}>
          <Text style={styles.label}>Booking For</Text>
          <Text style={styles.value}>{bookingFor === 'yourself' ? 'Yourself' : 'Another Person'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Last Name</Text>
          <Text style={styles.value}>{lastName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>First Name</Text>
          <Text style={styles.value}>{firstName}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Age</Text>
          <Text style={styles.value}>{age}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.label}>Gender</Text>
          <Text style={styles.value}>{gender}</Text>
        </View>
        
        <View style={styles.detailRow}>
          <Text style={styles.label}>Reason</Text>
          <Text style={styles.value}>{reason}</Text>
        </View>
      </View>

      <TouchableOpacity
        style={styles.doneButton}
        onPress={() => router.replace('/')}
      >
        <Text style={styles.doneButtonText}>Done</Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 30,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subHeaderText: {
    fontSize: 16,
    color: '#666',
    marginTop: 5,
  },
  card: {
    backgroundColor: '#f5f5f5',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
  },
  iconContainer: {
    alignItems: 'center',
    marginBottom: 15,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    paddingBottom: 5,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  label: {
    fontWeight: 'bold',
    color: '#555',
  },
  value: {
    color: '#333',
    maxWidth: '60%',
    textAlign: 'right',
  },
  divider: {
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
    marginVertical: 10,
  },
  doneButton: {
    backgroundColor: '#1E88E5',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 20,
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});