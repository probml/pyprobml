# demo
def f(x):
    if (x==1):
        return 'ok'
    else:
        return 'Error'
  
result = f(2)
if not(result == 'ok'):
    print('failure {}'.format(result))
